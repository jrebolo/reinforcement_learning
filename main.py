import logging
from src.agents.simple_cartpole import SimpleCartPole
import matplotlib.pyplot as plt

def main():
    # Create agent with desired logging level
    agent = SimpleCartPole()
    
    # Train
    agent.train(episodes=200)
    
    # Demonstrate
    agent.demonstrate()

if __name__ == "__main__":
    main()