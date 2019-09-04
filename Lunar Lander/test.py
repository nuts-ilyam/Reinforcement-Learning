import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import time
from model import size_layer
from model import Net

softmax = nn.Softmax(dim=1)
env = gym.make('LunarLander-v2')
# env.seed(1)

model = Net(8, size_layer, 4)
# best ones
model.load_state_dict(
    torch.load('C:\\Users\\Nuts\\PycharmProjects\\Lunar Lander without discounted rewards\\models\\new\\best_model4'))
# 118,  161
# model.load_state_dict(torch.load('C:\\Users\\Nuts\\PycharmProjects\\Lunar Lander without discounted rewards\\models\\best model173'))
start = time.clock()
model.eval()
state = env.reset()
total_reward = 0
# last_actions=[0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0]
for i in range(5000):
    env.render()
    state_tensor = torch.FloatTensor([state])
    actions_prob = softmax(model(state_tensor))
    action = actions_prob.argmax().numpy()
    state, reward, done, info = env.step(action)
    total_reward += reward
    print(action, reward)
    if done:
        print("TOTAL:", total_reward)
        break
