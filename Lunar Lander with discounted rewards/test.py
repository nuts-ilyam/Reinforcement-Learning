import torch
import gym
import torch.nn as nn
from model import Net
from model import size_layer

softmax = nn.Softmax(dim=1)
env = gym.make('LunarLander-v2')
# env.seed(1)

model = Net(8, size_layer, 4)
model.load_state_dict(torch.load('C:\\Users\\Nuts\\PycharmProjects\\LunarLander\\\models\\net_parameters4'))
# 4 - BEST
model.eval()
state = env.reset()
total_reward = 0
for i in range(5000):
    env.render()
    state_tensor = torch.FloatTensor([state])
    actions_prob = softmax(model(state_tensor))
    action = actions_prob.argmax().numpy()
    print(action)
    state, reward, done, info = env.step(action)
    total_reward += reward
    print(reward)

    if done:
        print("total", total_reward)
        break
