import gym
import numpy as np
import random
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20

class Agent:
    def __init__(self, epsilon=0.1):
        self.env = gym.make(ENV_NAME, is_slippery=True)
        self.state = self.env.reset()
        self.state = self.state[0] if isinstance(self.state, tuple) else self.state
        self.epsilon = epsilon
        self.values = np.ones((self.env.observation_space.n, self.env.action_space.n)) * 0.1
    
    def get_state(self, state):
        return state[0] if isinstance(state, tuple) else state
    
    def sample_env(self):
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            _, action = self.best_value_and_action(self.state)
        new_state, reward, terminated, truncated, _ = self.env.step(action)
        if terminated or truncated:
            new_state = self.get_state(self.env.reset())
        old_state = self.get_state(self.state)
        self.state = new_state
        return old_state, action, reward, new_state

    def best_value_and_action(self, state):
        best_value = None
        best_action = None
        for action in range(self.env.action_space.n):
            q_value = self.values[state, action]
            if best_value is None or q_value > best_value:
                best_value = q_value
                best_action = action
        return best_value, best_action

    def value_update(self, state, action, reward, new_state):
        best_value, _ = self.best_value_and_action(new_state)
        q_value_difference = reward + GAMMA * best_value - self.values[state, action]
        self.values[state, action] += ALPHA * q_value_difference


    def play_episode(self, env):
        total_reward = 0.0
        state = self.get_state(env.reset())
        while True:
            _, best_action = self.best_value_and_action(state)
            new_state, reward, terminated, truncated, _ = env.step(best_action)
            total_reward += reward
            if terminated or truncated:
                break
            state = self.get_state(new_state)
        return total_reward

    def print_values(self):
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                print("State:", state, "Action:", action, "Value:", self.values[state, action])
            print()

    def print_policy(self):
        policy = {}
        for state in range(self.env.observation_space.n):
            _, best_action = self.best_value_and_action(state)
            policy[state] = best_action
            print("State:", state, "Best Action:", best_action)
        return policy

if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-q-learning")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        state, action, reward, new_state = agent.sample_env()
        agent.value_update(state, action, reward, new_state)

        cumulative_reward = 0.0
        for _ in range(TEST_EPISODES):
            cumulative_reward += agent.play_episode(test_env)
        cumulative_reward /= TEST_EPISODES
        writer.add_scalar("reward", cumulative_reward, iter_no)
        if cumulative_reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, cumulative_reward))
            best_reward = cumulative_reward
        if cumulative_reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()

    # Print the Q-values and extract/print the policy
    agent.print_values()
    agent.print_policy()