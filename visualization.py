import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def visualize_pandemic_state(env, episode, step):
    plt.figure(figsize=(15, 5))
    
    # Plot network state
    plt.subplot(121)
    pos = nx.spring_layout(env.graph)
    
    # Color nodes based on state (S: blue, I: red, R: green)
    colors = []
    for i in range(env.n_nodes):
        if env.states[i, 0] == 1:  # Susceptible
            colors.append('blue')
        elif env.states[i, 1] == 1:  # Infected
            colors.append('red')
        else:  # Recovered
            colors.append('green')
    
    nx.draw(env.graph, pos, node_color=colors, with_labels=True)
    plt.title(f'Network State (Episode {episode}, Step {step})')
    
    # Plot SIR curves
    plt.subplot(122)
    s_count = np.sum(env.states[:, 0])
    i_count = np.sum(env.states[:, 1])
    r_count = np.sum(env.states[:, 2])
    
    plt.bar(['Susceptible', 'Infected', 'Recovered'], 
            [s_count, i_count, r_count],
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
            # Visualize every 5 steps
            if step % 5 == 0:
                visualize_pandemic_state(env, episode, step)
            
            # Get action from policy
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = model(state_tensor).max(1)[1].item()
            
            # Take action
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            step += 1
            
            if done:
                visualize_pandemic_state(env, episode, step)
                print(f"Episode {episode} finished after {step} steps. Total reward: {total_reward}")
                break

# Example usage
env = PandemicEnv(n_nodes=20, edge_p=0.2)  # Smaller network for visualization
trained_model = train_dqn(env)
evaluate_policy(env, trained_model)