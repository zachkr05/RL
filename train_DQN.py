import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from pandemic_sim import TravelEnv  # Make sure this imports your custom environment

# -----------------------------
# 1. Define the Q-network
# -----------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
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
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
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
    """
    Epsilon-greedy action selection:
    - With probability epsilon, pick a random action
    - Otherwise, pick the action with the highest Q-value from the network
    """
    if random.random() < epsilon:
        return random.randrange(action_dim)
    else:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = q_net(state_t)
        return q_values.argmax(dim=1).item()


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
    """
    Trains a QNetwork on the given environment using a basic DQN approach.
    """

    # Get size of state and action from environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Set up device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate Q-network and target network
    q_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    # Optimizer
    optimizer = optim.Adam(q_net.parameters(), lr=lr)

    # Replay buffer
    replay_buffer = ReplayBuffer(max_size=20000)

    # Epsilon schedule
    def get_epsilon(step):
        """Linearly decays epsilon from epsilon_start to epsilon_end over epsilon_decay steps."""
        return max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (step / epsilon_decay))
    
    # Keep track of rewards for plotting or debugging
    episode_rewards = []

    global_step = 0
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            epsilon = get_epsilon(global_step)
            action = select_action(state, q_net, epsilon, action_dim, device)
            
            # Step environment
            next_state, reward, done, info = env.step(action)
            
            # Store transition in replay buffer
            replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            global_step += 1
            
            # If we have enough samples in replay buffer, do a training step
            if len(replay_buffer) > batch_size:
                # Sample a batch
                states_b, actions_b, rewards_b, next_states_b, dones_b = replay_buffer.sample(batch_size)
                
                # Convert to tensors
                states_b = torch.FloatTensor(states_b).to(device)
                actions_b = torch.LongTensor(actions_b).to(device)
                rewards_b = torch.FloatTensor(rewards_b).to(device)
                next_states_b = torch.FloatTensor(next_states_b).to(device)
                dones_b = torch.FloatTensor(dones_b).to(device)

                # Compute current Q-values
                q_values = q_net(states_b)
                # Gather the Q-value corresponding to each action in the batch
                q_values = q_values.gather(1, actions_b.unsqueeze(1)).squeeze(1)

                # Compute target Q-values using target network
                with torch.no_grad():
                    next_q_values = target_net(next_states_b).max(dim=1)[0]
                target_q = rewards_b + gamma * next_q_values * (1 - dones_b)

                # Compute loss
                loss = nn.MSELoss()(q_values, target_q)
                
                # Gradient descent
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Update target network periodically
            if global_step % target_update_interval == 0:
                target_net.load_state_dict(q_net.state_dict())

        episode_rewards.append(total_reward)
        
        # Print some debug info
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1}/{num_episodes}, Avg Reward (last 10): {avg_reward:.2f}, Epsilon: {epsilon:.2f}")

    return episode_rewards, q_net


# -----------------------------
# 5. Main script entry point
# -----------------------------
if __name__ == "__main__":
    # Create the environment
    env = TravelEnv()
    
    # Train DQN
    rewards, trained_q_net = train_q_network(
        env,
        num_episodes=200,       # Increase if you want longer training
        batch_size=64,
        gamma=0.99,
        lr=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=500,
        target_update_interval=50
    )
    
    print("Training finished!")
    print(f"Final 10-episode average reward: {np.mean(rewards[-10:]):.2f}")

    # Optionally, watch a few episodes with the trained agent
    test_episodes = 5
    for ep in range(test_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            # Inference (no epsilon-greedy, use best action)
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = trained_q_net(state_t)
            action = q_values.argmax(dim=1).item()

            # Step
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state
        
        print(f"[Test Episode {ep+1}] Total Reward: {total_reward:.2f}")
