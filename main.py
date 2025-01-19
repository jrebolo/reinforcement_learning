from src.agents.simple_cartpole import SimpleCartPole
from src.utils.parameter_tuner import CartPoleTuner
from src.utils.vizualizer_tuner import TuningVisualizer

import glob
import os

def main():
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

    # Get the most recent results file
    results_files = glob.glob('tuning_results/cartpole_tuning_*.json')
    if not results_files:
        print("No results files found!")
        return
        
    latest_results = max(results_files, key=os.path.getctime)
    print(f"Visualizing results from: {latest_results}")
    
    # Create visualizer and generate all plots
    visualizer = TuningVisualizer(latest_results)
    visualizer.create_all_plots()


if __name__ == "__main__":
    main()