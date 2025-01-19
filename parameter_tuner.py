import itertools
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import logging
from dataclasses import dataclass
from typing import List, Tuple
import json
import os
from datetime import datetime
from src.config.logging_config import setup_logger

@dataclass
class ParameterConfig:
    learning_rate: float
    discount: float
    epsilon_decay: float
    n_buckets: Tuple[int, ...]
    
    def to_dict(self):
        return {
            'learning_rate': self.learning_rate,
            'discount': self.discount,
            'epsilon_decay': self.epsilon_decay,
            'n_buckets': self.n_buckets
        }

class CartPoleTuner:
    def __init__(self, agent_class, n_trials=3, episodes=200,log_level=logging.INFO):
        self.agent_class = agent_class
        self.n_trials = n_trials
        self.episodes = episodes
        self.logger = setup_logger('CartPoleTuner', log_level)
        
        # Define parameter ranges to search
        self.param_grid = {
            'learning_rate': [0.1, 0.01, 0.001],
            'discount': [0.9, 0.95, 0.99],
            'epsilon_decay': [0.99, 0.995, 0.999],
            'n_buckets': [
                (3, 3, 6, 6),
                (4, 4, 8, 8),
                (6, 6, 12, 12)
            ]
        }
        
    def evaluate_parameters(self, params: ParameterConfig) -> float:
        """Run multiple trials with given parameters and return average performance"""
        trial_rewards = []
        
        for trial in range(self.n_trials):
            # Create agent with current parameters
            agent = self.agent_class(
                learning_rate=params.learning_rate,
                discount=params.discount,
                epsilon_decay=params.epsilon_decay,
                n_buckets=params.n_buckets,
                log_level=logging.WARNING  # Reduce logging during tuning
            )
            
            # Train agent
            agent.train(episodes=self.episodes)
            
            # Calculate performance metric (average of last 50 episodes)
            final_performance = np.mean(agent.reward_history[-50:])
            trial_rewards.append(final_performance)
        
        avg_reward = np.mean(trial_rewards)
        std_reward = np.std(trial_rewards)
        
        self.logger.info(f"Parameters {params} - Avg Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        return avg_reward

    def _evaluate_params_wrapper(self, params):
        """Wrapper function for parallel processing"""
        config = ParameterConfig(*params)
        return (config, self.evaluate_parameters(config))

    def run_grid_search(self, n_workers=4):
        """Perform grid search over parameter combinations"""
        self.logger.info("Starting grid search...")
        
        # Generate all parameter combinations
        param_combinations = list(itertools.product(
            self.param_grid['learning_rate'],
            self.param_grid['discount'],
            self.param_grid['epsilon_decay'],
            self.param_grid['n_buckets']
        ))
        
        # Run evaluations in parallel
        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for config, avg_reward in executor.map(self._evaluate_params_wrapper, param_combinations):
                results.append((config, avg_reward))
        
        # Sort results by performance
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Save results
        self.save_results(results)
        
        return results
    
    def save_results(self, results):
        """Save tuning results to a file"""
        if not os.path.exists('tuning_results'):
            os.makedirs('tuning_results')
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'tuning_results/cartpole_tuning_{timestamp}.json'
        
        results_dict = {
            'timestamp': timestamp,
            'n_trials': self.n_trials,
            'episodes': self.episodes,
            'results': [
                {
                    'parameters': config.to_dict(),
                    'average_reward': float(avg_reward)
                }
                for config, avg_reward in results
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=4)
        
        self.logger.info(f"Results saved to {filename}")

def main():
    from src.agents.simple_cartpole import SimpleCartPole
        
    # Create and run tuner
    tuner = CartPoleTuner(SimpleCartPole)
    results = tuner.run_grid_search()
    
    # Print best parameters
    best_config, best_reward = results[0]
    tuner.logger.info("\nBest parameters found:")
    tuner.logger.info(f"Learning rate: {best_config.learning_rate}")
    tuner.logger.info(f"Discount factor: {best_config.discount}")
    tuner.logger.info(f"Epsilon decay: {best_config.epsilon_decay}")
    tuner.logger.info(f"Number of buckets: {best_config.n_buckets}")
    tuner.logger.info(f"Average reward: {best_reward:.2f}")

if __name__ == "__main__":
    main()