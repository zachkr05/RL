import numpy as np
import networkx as nx
import gym
from gym import spaces

class TravelEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # city data
        self.city_data = {
            0: {"name": "NYC", "pop": 8400000, "beds": 23000},
            1: {"name": "Buffalo", "pop": 278000, "beds": 1800},
            2: {"name": "Rochester", "pop": 211000, "beds": 1500},
            3: {"name": "Syracuse", "pop": 148000, "beds": 1000},
            4: {"name": "Albany", "pop": 99000, "beds": 800}
        }
        self.n_cities = len(self.city_data)
        
        # disease params
        self.mean_beta = 0.3  # Average infection rate
        self.gamma = 0.1      # Recovery rate

        # intialize the graph
        self.graph = nx.Graph()

        # fully conneced graph (i.e each node connects to all other nodes)
        self.edges = [
            (0, 1, 372), (0, 2, 450), (0, 3, 500), (0, 4, 157),
            (1, 2, 75), (1, 3, 140), (1, 4, 300),
            (2, 3, 88), (2, 4, 200), (3, 4, 147)
        ]

        # add the edges and their corresponding weights and 
        for edge in self.edges:
            self.graph.add_edge(edge[0], edge[1], weight=edge[2])
        
        # Action space: lockdowns + vaccinations
        self.action_space = spaces.Discrete(len(self.edges) + self.n_cities)

        # what info the agent can receive
        # box = continuous 
        # low = 0, high = 1, --> each value will lie between 0 and 1 
        # shape = the  observations is a 1D array of size self.n_cities * 3 i.e there are 4 possible features to observe (infection percentage, population , and recovered)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.n_cities * 3,), dtype=np.float32  
        )

    def get_infection_rate(self):
        """Generate a Poisson-distributed infection rate."""
        poisson_lambda = self.mean_beta * 10  # Scale lambda to ensure reasonable range
        beta = np.random.poisson(poisson_lambda) / 10  # Scale back to original range
        return min(beta, 1.0)  # Cap infection rate at 1.0


if __name__ == "__main__":
    # Example usage:
    env = TravelEnv()
    print("Generated infection rate (beta):", env.get_infection_rate())
