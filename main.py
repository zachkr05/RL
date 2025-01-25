import torch
import numpy as np
import matplotlib.pyplot as plt
from pandemic_sim import TravelEnv

class QNetwork(torch.nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, act_dim)
        )
    
    def forward(self, x):
        return self.net(x)

def run_trained_agent(num_episodes=5, max_steps=1000, model_path="q_network.pth"):
    # Initialize environment
    env = TravelEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    # Load trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_net = QNetwork(obs_dim, act_dim).to(device)
    q_net.load_state_dict(torch.load(model_path, map_location=device))
    q_net.eval()

    # Initialize visualization
    plt.figure(figsize=(15, 10))
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        # Tracking variables
        infections = {city: [] for city in range(env.n_cities)}
        budgets = []
        actions = []
        rewards = []

        print(f"\n=== Episode {episode+1} ===")
        
        while not done and step < max_steps:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # Get Q-values from network
            with torch.no_grad():
                q_values = q_net(state_tensor)
            
            # Choose best action
            action = q_values.argmax().item()
            
            # Take step in environment
            next_state, reward, done, _ = env.step(action)
            
            # Store metrics
            for city in range(env.n_cities):
                infections[city].append(state[1 + city*5])  # Infection index in observation
            budgets.append(env.budget)
            actions.append(action)
            rewards.append(reward)
            
            # Update state
            state = next_state
            step += 1
            
            if done:
                break

        # Visualization
        plt.clf()
        
        # Infection Plot
        plt.subplot(2, 2, 1)
        for city, inf_data in infections.items():
            plt.plot(inf_data, label=f'City {city}')
        plt.title('Infection Trends')
        plt.xlabel('Steps')
        plt.ylabel('Infected Population')
        plt.legend()
        
        # Budget Plot
        plt.subplot(2, 2, 2)
        plt.plot(budgets)
        plt.title('Budget Over Time')
        plt.xlabel('Steps')
        plt.ylabel('Remaining Budget')
        
        # Action Distribution
        plt.subplot(2, 2, 3)
        action_counts = np.bincount(actions, minlength=act_dim)
        plt.bar(range(act_dim), action_counts)
        plt.title('Action Distribution')
        plt.xlabel('Action ID')
        plt.ylabel('Count')
        
        # Reward Tracking
        plt.subplot(2, 2, 4)
        cumulative_rewards = np.cumsum(rewards)
        plt.plot(cumulative_rewards)
        plt.title('Cumulative Reward')
        plt.xlabel('Steps')
        plt.ylabel('Total Reward')

        plt.tight_layout()
        plt.savefig(f'episode_{episode+1}_performance.png')
        plt.pause(0.1)

        # Episode summary
        print(f"\nEpisode {episode+1} Summary:")
        print(f"Total Steps: {step}")
        print(f"Final Budget: {env.budget:.2f}")
        print(f"Total Infections: {sum(infections[city][-1] for city in range(env.n_cities))}")
        print(f"Total Reward: {sum(rewards):.2f}")
        print(f"Most Common Action: {np.argmax(action_counts)} (Count: {np.max(action_counts)})")

if __name__ == "__main__":
    run_trained_agent(
        num_episodes=3,
        max_steps=20,
        model_path="q_network.pth"
    )