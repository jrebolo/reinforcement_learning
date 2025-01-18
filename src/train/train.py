import gymnasium as gym
import numpy as np
from ..agents.cart_pole_agent import CartPoleAgent


def train_agent():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = CartPoleAgent(state_size, action_size)
    episodes = 500
    
    for episode in range(episodes):
        state, _ = env.reset()  # Gymnasium returns (state, info)
        total_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)  # Gymnasium returns (state, reward, terminated, truncated, info)
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            agent.train()
            
        if episode % agent.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

if __name__ == "__main__":
    train_agent()