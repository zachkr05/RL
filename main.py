from stable_baselines3 import DQN
from pandemic_sim import TravelEnv  # Import your custom environment
from scipy.io import savemat  # To save data in MATLAB format

def evaluate_dqn(env, model, num_episodes=100):
    """
    Run the trained model for a given number of episodes
    and print out the total rewards and step counts.
    """
    max_steps = 1000  # Maximum steps per episode
    rewards = []  # List to store rewards per episode
    steps = []    # List to store steps per episode

    for episode in range(num_episodes):
        print(f"Starting Episode {episode + 1}")
        obs = env.reset()
        done = False
        total_reward = 0
        step_count = 0  # To count steps per episode

        while not done:
            # Use the trained DQN model to predict actions
            action, _ = model.predict(obs, deterministic=True)

            # Step through the environment
            obs, reward, done, info = env.step(action)

            # Accumulate the reward
            total_reward += reward
            step_count += 1

            # End the episode if steps exceed 100
            if step_count >= max_steps:
                print(f"Step limit of {max_steps} reached. Ending episode early.")
                break

        rewards.append(total_reward)
        steps.append(step_count)
        print(f"Episode {episode + 1} ended with Total Reward: {total_reward}")
        print(f"Steps in Episode {episode + 1}: {step_count}")
        print("=" * 50)

    # Save data to a MATLAB file
    savemat("dqn_learning_data.mat", {"rewards": rewards, "steps": steps})
    print("Data saved to dqn_learning_data.mat")


if __name__ == "__main__":
    # Create the env
    test_env = TravelEnv()

    # Load the saved model
    loaded_model = DQN.load("dqn_travel_env")

    # Evaluate
    evaluate_dqn(test_env, loaded_model, num_episodes=10)
