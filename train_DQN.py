import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from pandemic_sim import TravelEnv  # Make sure this imports your custom environment
import logging
import sys
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f'dqn_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# -----------------------------
# 1. Define the Q-network
# -----------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        logging.info(f"Initialized QNetwork with dims: state={state_dim}, action={action_dim}, hidden={hidden_dim}")
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# -----------------------------
# 2. Define a Replay Buffer
# -----------------------------
class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)
        logging.info(f"Initialized ReplayBuffer with max_size={max_size}")
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) % 1000 == 0:
            logging.info(f"ReplayBuffer size: {len(self.buffer)}")
        
    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states),
                np.array(actions),
                np.array(rewards),
                np.array(next_states),
                np.array(dones, dtype=np.float32))
        
    def __len__(self):
        return len(self.buffer)


# -----------------------------
# 3. Utility function to choose an action (epsilon-greedy)
# -----------------------------
def select_action(state, q_net, epsilon, action_dim, device):
    if random.random() < epsilon:
        action = random.randrange(action_dim)
        logging.debug(f"Random action selected: {action} (epsilon={epsilon:.3f})")
        return action
    
    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = q_net(state_t)
    action = q_values.argmax(dim=1).item()
    logging.debug(f"Q-network action selected: {action} (epsilon={epsilon:.3f})")
    return action


# -----------------------------
# 4. Training loop for the Q-network
# -----------------------------
def train_q_network(
    env,
    num_episodes=500,
    batch_size=64,
    gamma=0.99,
    lr=1e-3,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=500,
    target_update_interval=50
):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.info(f"""Training configuration:
    Device: {device}
    State dim: {state_dim}
    Action dim: {action_dim}
    Episodes: {num_episodes}
    Batch size: {batch_size}
    Learning rate: {lr}
    Epsilon: {epsilon_start} → {epsilon_end} over {epsilon_decay} steps
    Target update interval: {target_update_interval}
    """)

    q_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(max_size=20000)
    
    def get_epsilon(step):
        return max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (step / epsilon_decay))
    
    episode_rewards = []
    global_step = 0
    best_avg_reward = float('-inf')
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        episode_loss = []
        done = False
        step = 0
        
        logging.info(f"\nStarting Episode {episode+1}")
        
        while not done:
            epsilon = get_epsilon(global_step)
            action = select_action(state, q_net, epsilon, action_dim, device)
            
            next_state, reward, done, info = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            
            logging.debug(f"""Step {step}:
            Action: {action}
            Reward: {reward:.2f}
            Done: {done}
            Info: {info}""")
            
            state = next_state
            total_reward += reward
            global_step += 1
            step += 1
            
            if len(replay_buffer) > batch_size:
                states_b, actions_b, rewards_b, next_states_b, dones_b = replay_buffer.sample(batch_size)
                
                states_b = torch.FloatTensor(states_b).to(device)
                actions_b = torch.LongTensor(actions_b).to(device)
                rewards_b = torch.FloatTensor(rewards_b).to(device)
                next_states_b = torch.FloatTensor(next_states_b).to(device)
                dones_b = torch.FloatTensor(dones_b).to(device)

                q_values = q_net(states_b).gather(1, actions_b.unsqueeze(1)).squeeze(1)
                
                with torch.no_grad():
                    next_q_values = target_net(next_states_b).max(dim=1)[0]
                target_q = rewards_b + gamma * next_q_values * (1 - dones_b)

                loss = nn.MSELoss()(q_values, target_q)
                episode_loss.append(loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if global_step % target_update_interval == 0:
                    target_net.load_state_dict(q_net.state_dict())
                    logging.info("Updated target network")

        episode_rewards.append(total_reward)
        avg_reward = np.mean(episode_rewards[-10:])
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        
        logging.info(f"""Episode {episode+1} Summary:
        Total Reward: {total_reward:.2f}
        Average Reward (last 10): {avg_reward:.2f}
        Average Loss: {avg_loss:.4f}
        Steps: {step}
        Epsilon: {epsilon:.3f}""")
        
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            torch.save(q_net.state_dict(), f'best_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
            logging.info(f"New best average reward: {best_avg_reward:.2f} - Model saved")

    return episode_rewards, q_net


# -----------------------------
# 5. Main script entry point
# -----------------------------
if __name__ == "__main__":
    logging.info("Starting DQN training script")
    env = TravelEnv()
    
    rewards, trained_q_net = train_q_network(
        env,
        num_episodes=200,
        batch_size=64,
        gamma=0.99,
        lr=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=500,
        target_update_interval=50
    )
    
    logging.info("\nTraining completed!")
    logging.info(f"Final 10-episode average reward: {np.mean(rewards[-10:]):.2f}")
    
    torch.save(trained_q_net.state_dict(), f'final_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
    logging.info("Final model saved")
