# visualization.py
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import os

class TuningVisualizer:
    def __init__(self, results_path, agent=None):
        """
        Initialize visualizer with path to results file
        
        Args:
            results_path (str): Path to the JSON results file
        """
        with open(results_path, 'r') as f:
            self.raw_data = json.load(f)
            
        # Convert results to DataFrame for easier plotting
        self.df = pd.DataFrame([
            {
                'learning_rate': result['parameters']['learning_rate'],
                'discount': result['parameters']['discount'],
                'epsilon_decay': result['parameters']['epsilon_decay'],
                'n_buckets': str(result['parameters']['n_buckets']),  # Convert tuple to string for plotting
                'average_reward': result['average_reward']
            }
            for result in self.raw_data['results']
        ])

        self.agent = agent
        
    def plot_parameter_distributions(self, save_path=None):
        """Create violin plots showing reward distributions for each parameter"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Parameter Performance Distributions', fontsize=16)
        
        # Plot for each parameter
        params = ['learning_rate', 'discount', 'epsilon_decay', 'n_buckets']
        for ax, param in zip(axes.flat, params):
            sns.violinplot(data=self.df, x=param, y='average_reward', ax=ax)
            ax.set_title(f'Impact of {param}')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_parameter_heatmaps(self, save_path=None):
        """Create heatmaps showing interactions between parameters"""
        params = ['learning_rate', 'discount', 'epsilon_decay']
        n_params = len(params)
        fig, axes = plt.subplots(n_params, n_params, figsize=(15, 15))
        fig.suptitle('Parameter Interaction Heatmaps', fontsize=16)
        
        for i, param1 in enumerate(params):
            for j, param2 in enumerate(params):
                ax = axes[i, j]
                if i != j:
                    # Create heatmap for parameter interactions
                    pivot = pd.pivot_table(
                        self.df,
                        values='average_reward',
                        index=param1,
                        columns=param2,
                        aggfunc='mean'
                    )
                    sns.heatmap(pivot, ax=ax, cmap='viridis', annot=True, fmt='.1f')
                    ax.set_title(f'{param1} vs {param2}')
                else:
                    # Show parameter distribution on diagonal
                    ax.hist(self.df[param1])
                    ax.set_title(f'{param1} distribution')
                
                ax.tick_params(axis='x', rotation=45)
                ax.tick_params(axis='y', rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_top_configurations(self, top_n=10, save_path=None):
        """Plot the top N performing parameter configurations"""
        try:
            # Ensure average_reward is numeric
            self.df['average_reward'] = pd.to_numeric(self.df['average_reward'], errors='raise')
            top_configs = self.df.nlargest(top_n, 'average_reward')
            
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(range(len(top_configs)), top_configs['average_reward'])
            
            # Add parameter annotations to bars
            for idx, bar in enumerate(bars):
                config = top_configs.iloc[idx]
                annotation = f"lr={config['learning_rate']:.3f}\nγ={config['discount']:.3f}\nε={config['epsilon_decay']:.3f}\nb={config['n_buckets']}"
                ax.text(idx, bar.get_height(), annotation,
                       ha='center', va='bottom', rotation=0,
                       bbox=dict(facecolor='white', alpha=0.7))
            
            ax.set_title('Top Performing Configurations')
            ax.set_xlabel('Configuration Rank')
            ax.set_ylabel('Average Reward')
            
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path)
            plt.show()
            
        except Exception as e:
            print(f"Error plotting top configurations: {str(e)}")
            print("DataFrame head:")
            print(self.df.head())
            print("\nDataFrame info:")
            print(self.df.info())
        
    def plot_learning_rate_vs_discount(self, save_path=None):
        """Create a scatter plot of learning rate vs discount factor"""
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(self.df['learning_rate'], 
                            self.df['discount'],
                            c=self.df['average_reward'],
                            s=100, cmap='viridis')
        
        plt.colorbar(scatter, label='Average Reward')
        plt.xlabel('Learning Rate')
        plt.ylabel('Discount Factor')
        plt.title('Learning Rate vs Discount Factor Performance')
        
        # Add annotations for top 3 points
        top_3 = self.df.nlargest(3, 'average_reward')
        for _, point in top_3.iterrows():
            plt.annotate(f'Reward: {point["average_reward"]:.1f}',
                        (point['learning_rate'], point['discount']),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(facecolor='white', alpha=0.7))
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

    

    def create_all_plots(self, output_dir='tuning_plots'):
        """Generate and save all visualization plots"""
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate all plots
        self.plot_parameter_distributions(f'{output_dir}/parameter_distributions.png')
        self.plot_parameter_heatmaps(f'{output_dir}/parameter_heatmaps.png')
        self.plot_top_configurations(f'{output_dir}/top_configurations.png')
        self.plot_learning_rate_vs_discount(f'{output_dir}/learning_rate_vs_discount.png')


def main():
    # Example usage
    import glob
    from src.agents.simple_cartpole import SimpleCartPole

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