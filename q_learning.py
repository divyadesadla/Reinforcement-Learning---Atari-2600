
from environment import MountainCar
import numpy as np
import sys
import random


def lst_from_dict(S, state):
    S *= 0
    for key in state.keys():
        S[key] = state[key]


def q_learning(env, weight_table, bias, max_iterations, episodes, epsilon, gamma, learning_rate):

    rewards = 0

    S = np.zeros((weight_table.shape[0], 1))
    initial_state = env.reset()
    lst_from_dict(S, initial_state)

    for j in range(max_iterations):

        rand = np.random.uniform(0, 1)
        if rand < epsilon:
            action = np.random.randint(3)
        else:
            q_val = np.dot(S.T, weight_table) + bias
            action = np.argmax(q_val)

        gradient = np.zeros((weight_table.shape))
        gradient[:, action] = S[:, 0]

        q_now = np.dot(S.T, weight_table[:, action].reshape(
            (weight_table.shape[0], 1))) + bias

        state_prime, reward, done = env.step(action)
        lst_from_dict(S, state_prime)

        q_val2 = np.dot(S.T, weight_table) + bias
        action_next = np.argmax(q_val2)

        q_val_next = np.dot(S.T, weight_table[:, action_next].reshape(
            (weight_table.shape[0], 1))) + bias
        # next_max = np.max(q_val_next)

        weight_table = weight_table - learning_rate * \
            (q_now - (reward + gamma * q_val_next)) * gradient

        bias = bias - learning_rate * (q_now - (reward + gamma * q_val_next))
        # state = state_prime
        rewards += reward

        if done:
            break
    return weight_table, bias, rewards


def running(env, weight_table, bias, max_iterations, episodes, epsilon, gamma, learning_rate):
    bias = 0
    total_reward_str = ''

    for i in range(episodes):
        weight_table, bias, rewards = q_learning(
            env, weight_table, bias, max_iterations, episodes, epsilon, gamma, learning_rate)
        total_reward_str += str(rewards) + '\n'
    env.close()
    return weight_table, bias, total_reward_str


if __name__ == "__main__":
    mode = sys.argv[1]
    weight_out = sys.argv[2]
    returns_out = sys.argv[3]
    episodes = int(sys.argv[4])
    max_iterations = int(sys.argv[5])
    epsilon = float(sys.argv[6])
    gamma = float(sys.argv[7])
    learning_rate = float(sys.argv[8])

    # mode = 'tile'
    # episodes = 20
    # max_iterations = 200
    # epsilon = 0.05
    # gamma = 0.99
    # learning_rate = 0.00005
    # weight_out = 'tile_weight.out'
    # returns_out = 'tile_returns.out'

    env = MountainCar(mode)
    weight_table = np.zeros((env.state_space, env.action_space))
    bias = 0

    weight_table, bias, total_reward_str = running(
        env, weight_table, bias, max_iterations, episodes, epsilon, gamma, learning_rate)

    with open(returns_out, 'w') as y:
        y.write(total_reward_str)

    # print(bias)
    str_weight = str(bias[0][0]) + '\n'
    for k in weight_table:
        for kk in k:
            str_weight += str(kk) + '\n'

    with open(weight_out, 'w') as f:
        f.write(str_weight)
