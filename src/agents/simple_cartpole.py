import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
import logging

from src.config.logging_config import setup_logger

class SimpleCartPole:
    def __init__(self, log_level=logging.INFO):
        # Set up logging
        self.logger = setup_logger('CartPole', log_level)
        
        self.logger.info("Initializing CartPole environment")
        self.env = gym.make('CartPole-v1', render_mode='human')
        self.training_env = gym.make('CartPole-v1')
        
        # Discretize the continuous state space into buckets
        # position, velocity, angle, angular velocity
        self.n_buckets = (3, 3, 6, 6)
        self.state_bounds = [
            [-4.8, 4.8],
            [-5, 5],
            [-0.418, 0.418],
            [-5, 5]
        ]
        
        self.logger.info(f"State space discretization: {self.n_buckets} buckets")
        self.logger.info(f"State bounds: {self.state_bounds}")
        
        # Create Q-table
        self.q_table = np.zeros(self.n_buckets + (2,))
        
        # Learning parameters
        self.learning_rate = 0.1
        self.discount = 0.95
        self.epsilon = 1.0
        
        self.logger.info(f"Learning parameters - LR: {self.learning_rate}, Discount: {self.discount}, Initial epsilon: {self.epsilon}")
        self.logger.info(f"Q Table Dimentions: {self.q_table.shape}")
        # Training history
        self.reward_history = []
        self.epsilon_history = []

    def discretize_state(self, state):
        discrete_state = []
        for i, value in enumerate(state):
            scaled = (value - self.state_bounds[i][0]) / (self.state_bounds[i][1] - self.state_bounds[i][0])
            scaled = min(max(scaled, 0), 0.999)
            discrete_state.append(int(scaled * self.n_buckets[i]))
            
            if self.logger.level == logging.DEBUG:
                state_names = ['position', 'velocity', 'angle', 'angular velocity']
                self.logger.debug(f"{state_names[i]}: {value:.3f} → bucket {discrete_state[-1]}")
                
        return tuple(discrete_state)
    
    def select_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            action = self.training_env.action_space.sample()
            self.logger.debug(f"Exploring: Random action {action}")
            return action
        
        action = np.argmax(self.q_table[state])
        self.logger.debug(f"Exploiting: Chosen action {action}")
        return action
    
    def train(self, episodes=100):
        self.logger.info(f"Starting training for {episodes} episodes")
        
        for episode in range(episodes):
            state, _ = self.training_env.reset()
            discrete_state = self.discretize_state(state)
            total_reward = 0
            steps = 0
            done = False
            truncated = False
            
            self.logger.info(f"Starting episode {episode}")
            
            while not (done or truncated):
                action = self.select_action(discrete_state)
                next_state, reward, done, truncated, _ = self.training_env.step(action)
                next_discrete_state = self.discretize_state(next_state)
                total_reward += reward
                steps += 1
                
                # Q-value update
                old_value = self.q_table[discrete_state + (action,)]
                next_max = np.max(self.q_table[next_discrete_state])
                new_value = old_value + self.learning_rate * (reward + self.discount * next_max - old_value)
                self.q_table[discrete_state + (action,)] = new_value
                
                if self.logger.level == logging.DEBUG:
                    self.logger.debug(f"Step {steps}: Q-value update {old_value:.3f} → {new_value:.3f}")
                
                discrete_state = next_discrete_state
            
            # Record history
            self.reward_history.append(total_reward)
            self.epsilon_history.append(self.epsilon)
            
            # Decay epsilon
            self.epsilon = max(0.01, self.epsilon * 0.995)
            
            self.logger.info(f"Episode {episode} finished - Steps: {steps}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.3f}")
        
        self.logger.info("Training completed")
    
    def demonstrate(self, episodes=3):
        self.logger.info(f"Starting demonstration for {episodes} episodes")
        
        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0
            steps = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                discrete_state = self.discretize_state(state)
                action = self.select_action(discrete_state, training=False)
                state, reward, done, truncated, _ = self.env.step(action)
                total_reward += reward
                steps += 1
                time.sleep(0.01)
            
            self.logger.info(f"Demonstration episode {episode + 1} - Steps: {steps}, Total Reward: {total_reward}")
        
        self.env.close()
        self.logger.info("Demonstration completed")

# Run the training and visualization
if __name__ == "__main__":
    agent = SimpleCartPole()
    
    agent.train(episodes=1000)
        
    # Demonstrate the trained agent
    agent.demonstrate()