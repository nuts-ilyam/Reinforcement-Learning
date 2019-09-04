import gym
from collections import namedtuple
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import Net


def play_episode(env):
    states = []
    rewards = []
    actions = []

    state = env.reset()
    start = time.clock()
    # data for train
    for i in range(max_iter):
        state_tensor = torch.FloatTensor([state])
        actions_prob = softmax(net(state_tensor))
        # print(actions_prob, type(actions_prob))
        actions_prob = actions_prob.data.numpy()[0]
        # print(actions_prob, type(actions_prob))
        a = np.random.choice(len(actions_prob), p=actions_prob)
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


def learn(actions, states, rewards, threshold):
    actions, states = make_one_list(actions, states, rewards, threshold)
    optimizer.zero_grad()
    targets = torch.LongTensor(actions)
    outputs = net(torch.FloatTensor(states))
    loss = loss_function(outputs, targets)
    loss.backward()
    optimizer.step()
    return loss.item()


def make_one_list(actions, states, rewards, threshold):
    list_actions, list_states = [], []
    for acts, sts, rew in zip(actions, states, rewards):
        if rew > threshold:
            for a, s in zip(acts, sts):
                list_actions.append(a)
                list_states.append(s)

    return list_actions, list_states


env = gym.make('LunarLander-v2')
f = open('rewards.txt', 'w')
size_layer = 100
lr = 0.01
n_episodes = 100
n_epoch = 200
desired_rate = 150
max_iter = 5000
size_states = env.observation_space.shape[0]
size_actions = env.action_space.n

net = Net(size_states, size_layer, size_actions)

loss_function = nn.CrossEntropyLoss()

# optimizer = optim.SGD(params=net.parameters(), lr=lr)
optimizer = optim.Adam(params=net.parameters(), lr=lr)
softmax = nn.Softmax(dim=1)

for epoch in range(n_epoch):
    states, actions, rewards = [], [], []

    for episode in range(n_episodes):
        episode_return = play_episode(env)
        states.append(episode_return[0])
        actions.append(episode_return[1])
        total_reward = np.sum(episode_return[2])  # reward
        rewards.append(total_reward)

    threshold = np.percentile(rewards, 75)
    loss = learn(actions, states, rewards, threshold)

    mean_reward = np.mean(rewards)
    f.write(str(mean_reward) + '\n')
    print("%d:  reward_mean=%.1f, loss=%.3f, reward_threshold75=%.1f" % (
        epoch, mean_reward, loss, threshold))

    if mean_reward > desired_rate:
        torch.save(net.state_dict(),
                   'C:\\Users\\Nuts\\PycharmProjects\\Lunar Lander without discounted rewards\\models\\best model%d' % (
                       epoch))

torch.save(net.state_dict(),
           'C:\\Users\\Nuts\\PycharmProjects\\Lunar Lander without discounted rewards\\best model')
f.close()
