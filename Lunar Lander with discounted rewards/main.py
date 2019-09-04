import gym
import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import tqdm
import time
from model import Net


def buffer_append(episode_states, episode_actions, episode_rewards):
    part = 0.05
    idxs = np.random.choice(np.arange(len(episode_states)), size=int(len(episode_states) * part),
                            replace=False)
    buffer_states.extend([episode_states[i] for i in idxs])
    buffer_actions.extend([episode_actions[i] for i in idxs])
    buffer_rewards.extend([episode_rewards[i] for i in idxs])


def get_buffer_data(n=1):
    n = min(n, len(buffer_states))
    idxs = np.random.choice(np.arange(len(buffer_states)),
                            size=n,
                            replace=False)
    return [buffer_states[i] for i in idxs], \
           [buffer_actions[i] for i in idxs], \
           [buffer_rewards[i] for i in idxs]


def discount_reward(rewards, gamma):
    sum_ = 0
    discounted_rewards = np.zeros_like(rewards)
    for r in reversed(range(len(rewards))):  #
        sum_ = sum_ * gamma + (rewards[r])
        discounted_rewards[r] = sum_

    # mean_ = np.mean(discounted_rewards)
    # std_ = np.std(discounted_rewards)
    # discounted_rewards = (discounted_rewards - mean_) / std_
    return (discounted_rewards)


def play_episode(env):
    activation = nn.Softmax(dim=1)
    states = []
    rewards = []
    actions = []

    state = env.reset()
    start = time.clock()
    # data for train
    for i in range(max_iter):
        state_tensor = torch.FloatTensor([state])
        actions_prob = activation(net(state_tensor))
        actions_prob = actions_prob.detach().numpy()
        # print(actions_prob, type(actions_prob))
        a = np.random.choice(range(len(actions_prob.ravel())), p=actions_prob.ravel())
        # a = actions_prob.argmax().numpy()
        new_step = env.step(a)
        new_state = new_step[0]
        reward = new_step[1]
        done = new_step[2]

        rewards.append(reward)
        states.append(state)
        actions.append(int(a))
        end = time.clock()
        state = new_state
        if end - start > 100:
            print("Time over", end - start)
            done = True

        if done:
            break
    return states, actions, rewards


env = gym.make('LunarLander-v2')

size_layer = 100
lr = 0.001
gamma = 0.99
n_episodes = 1000
n_epoch = 5

max_iter = 5000
size_states = env.observation_space.shape[0]
size_actions = env.action_space.n

net = Net(size_states, size_layer, size_actions)
loss_function = nn.CrossEntropyLoss(reduce=False)
optimizer = optim.Adam(params=net.parameters(), lr=lr)
activation = nn.Softmax(dim=1)

desired_rate = 200

loss_list = []
total_reward = []
buffer_states = []
buffer_actions = []
buffer_rewards = []
threshold25 = -10000
threshold90 = -10000

for epoch in range(n_epoch):
    print("epoch: ", epoch)
    for i, episode in enumerate(range(n_episodes)):
        states, actions, rewards = play_episode(env)
        if (i + 1) % 100 == 0:
            threshold = np.percentile(total_reward, 75)
            # threshold25 = np.percentile(total_reward, 25)
            mean_reward = np.mean(total_reward)
            mean_loss = np.mean(loss_list)
            print("%d:  reward_mean=%.1f, loss=%.4f, reward_threshold75=%.1f" % (
                i // 100, mean_reward, mean_loss, threshold))

            if mean_reward > desired_rate:
                torch.save(net.state_dict(),
                           'C:\\Users\\Nuts\\PycharmProjects\\LunarLander\\models\\net_parameters%d' % (epoch))
            loss_list = []
            total_reward = []

        total = np.sum(rewards)
        total_reward.append(total)

        rewards = torch.FloatTensor(rewards)
        discounted_rewards = discount_reward(rewards, gamma)

        # train current episode
        optimizer.zero_grad()
        targets = torch.LongTensor(actions)
        outputs = net(torch.FloatTensor(states))
        loss = loss_function(outputs, targets)

        # loss * discounted reward
        loss = torch.mean(loss * torch.FloatTensor(discounted_rewards))
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

        # train buffer
        buffer_append(states, actions, discounted_rewards)
        optimizer.zero_grad()
        b_states, b_actions, b_rewards = get_buffer_data(100)
        targets = torch.LongTensor(b_actions)
        outputs = activation(net(torch.FloatTensor(b_states)))
        loss = loss_function(outputs, targets)
        loss = torch.mean(loss)
        loss.backward()
        optimizer.step()

torch.save(net.state_dict(), 'C:\\Users\\Nuts\\PycharmProjects\\LunarLander\\net_parameters')

# show
model = Net(8, size_layer, 4)
model.load_state_dict(torch.load('C:\\Users\\Nuts\\PycharmProjects\\LunarLander\\net_parameters'))
model.eval()
state = env.reset()
for i in range(5000):
    env.render()
    state_tensor = torch.FloatTensor([state])
    actions_prob = activation(model(state_tensor))
    action = actions_prob.argmax().numpy()
    state, reward, done, info = env.step(action)

    if done:
        break
