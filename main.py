import networkx as nx
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt

class PandemicEnv(gym.Env):
    def __init__(self, n_nodes=100, edge_p=0.1):
        super().__init__()
        self.n_nodes = n_nodes
        self.edge_p = edge_p
        self.beta = 0.3
        self.gamma = 0.1
        self.action_space = spaces.Discrete(n_nodes)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(n_nodes * 3,), dtype=np.float32
        )
        self.reset()
    
    def reset(self, seed=None):
        self.graph = nx.erdos_renyi_graph(self.n_nodes, self.edge_p)
        self.states = np.zeros((self.n_nodes, 3))
        self.states[:, 0] = 1
        patient_zero = np.random.randint(self.n_nodes)
        self.states[patient_zero] = [0, 1, 0]
        return self._get_observation(), {}
    
    def _get_observation(self):
        return self.states.flatten()
    
    def step(self, action):
        if self.states[action, 0] == 1:
            self.states[action] = [0, 0, 1]
        
        new_states = self.states.copy()
        for node in range(self.n_nodes):
            if self.states[node, 1] == 1:
                for neighbor in self.graph.neighbors(node):
                    if self.states[neighbor, 0] == 1:
                        if np.random.random() < self.beta:
                            new_states[neighbor] = [0, 1, 0]
                if np.random.random() < self.gamma:
                    new_states[node] = [0, 0, 1]
        
        self.states = new_states
        reward = -(np.sum(self.states[:, 1]))
        done = np.sum(self.states[:, 1]) == 0
        return self._get_observation(), reward, done, False, {}

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done))
    
    def __len__(self):
        return len(self.buffer)

def train_dqn(env, episodes=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
    target_net = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters())
    memory = ReplayBuffer()
    
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = policy_net(state_tensor).max(1)[1].item()
            
            next_state, reward, done, _, _ = env.step(action)
            memory.push(state, action, reward, next_state, done)
            total_reward += reward
            
            if len(memory) >= batch_size:
                states, actions, rewards, next_states, dones = memory.sample(batch_size)
                
                state_batch = torch.FloatTensor(states).to(device)
                action_batch = torch.LongTensor(actions).to(device)
                reward_batch = torch.FloatTensor(rewards).to(device)
                next_state_batch = torch.FloatTensor(next_states).to(device)
                done_batch = torch.FloatTensor(dones).to(device)
                
                current_q_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
                next_q_values = target_net(next_state_batch).max(1)[0].detach()
                target_q_values = reward_batch + gamma * next_q_values * (1 - done_batch)
                
                loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if done:
                break
            state = next_state
        
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")
    
    return policy_net

def visualize_pandemic_state(env, episode, step):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(121)
    pos = nx.spring_layout(env.graph)
    colors = ['blue' if s[0] == 1 else 'red' if s[1] == 1 else 'green' for s in env.states]
    nx.draw(env.graph, pos, node_color=colors, with_labels=True)
    plt.title(f'Network State (Episode {episode}, Step {step})')
    
    plt.subplot(122)
    counts = [np.sum(env.states[:, i]) for i in range(3)]
    plt.bar(['Susceptible', 'Infected', 'Recovered'], counts,
            color=['blue', 'red', 'green'])
    plt.title('Population State')
    plt.ylabel('Number of Individuals')
    
    plt.tight_layout()
    plt.show()

def evaluate_policy(env, model, episodes=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for episode in range(episodes):
        state, _ = env.reset()
        step = 0
        total_reward = 0
        
        while True:
            if step % 5 == 0:
                visualize_pandemic_state(env, episode, step)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = model(state_tensor).max(1)[1].item()
            
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            step += 1
            
            if done:
                visualize_pandemic_state(env, episode, step)
                print(f"Episode {episode} finished after {step} steps. Total reward: {total_reward}")
                break

if __name__ == "__main__":
    # Create and train
    env = PandemicEnv(n_nodes=20, edge_p=0.2)
    trained_model = train_dqn(env, episodes=100)
    
    # Evaluate
    evaluate_policy(env, trained_model, episodes=3)