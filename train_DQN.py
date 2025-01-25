import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from pandemic_sim import TravelEnv  # Assuming your file is named pandemic_sim.py

def main():
    # Instantiate the environment
    env = TravelEnv()

    # Optional: Wrap in a DummyVecEnv if you want parallelization support
    env = DummyVecEnv([lambda: env])

    # Create the DQN model
    # "MlpPolicy" means a simple multi-layer perceptron neural network
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=100000,  # Larger buffer size for more training data
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,  # Start with high exploration
        exploration_final_eps=0.05,   # End with low exploration
        verbose=1
    )

    # Train the model for a longer duration
    model.learn(total_timesteps=100000)  # Train for 100k steps instead of 20k

    # Save the model
    model.save("dqn_travel_env")

    print("Training finished and model saved!")

if __name__ == "__main__":
    main()
