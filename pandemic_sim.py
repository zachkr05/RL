import numpy as np
import networkx as nx
import gym
from gym import spaces
from CityManager import CityManager

class TravelEnv(gym.Env):
    """Gym environment for simulating pandemic spread across cities with population-based infections."""
    
    BASE_BUDGET_PER_CITY = 100000
    BASE_LOCKDOWN_COST = 5000
    BASE_VACCINATION_COST = 10000
    MEAN_BETA = 0.3
    GAMMA = 0.1
    TRANSMISSION_THRESHOLD = 0.1  # Minimum infection percentage for transmission
    
    def __init__(self):
        super().__init__()
        self._initialize_city_data()
        self._setup_graph()
        self._setup_spaces()
        self._initialize_states()
        self._calculate_budget()

    def _initialize_city_data(self):
        self.city_manager = CityManager()
        self.n_cities = self.city_manager.n_cities
        self.city_populations = {
            city_id: self.city_manager.get_city_data(city_id)["population"]
            for city_id in range(self.n_cities)
        }

        self.marketing_cleanliness_factors = {
            city_id: 0.2  # For example, each city can have a different factor
            for city_id in range(self.n_cities)
        }


    def _setup_graph(self):
        self.graph = nx.Graph()
        self.edges = [
            (0, 1, 372), (0, 2, 450), (0, 3, 500), (0, 4, 157),
            (1, 2, 75), (1, 3, 140), (1, 4, 300),
            (2, 3, 88), (2, 4, 200), (3, 4, 147)
        ]
        for edge in self.edges:
            self.graph.add_edge(edge[0], edge[1], weight=edge[2])

    def _setup_spaces(self):
        self.action_space = spaces.Discrete(len(self.edges) + self.n_cities)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.n_cities * 3,), dtype=np.float32
        )

    def _initialize_states(self):
        """Initialize city states with actual infected population counts."""
        self.city_states = {}
        for city_id in range(self.n_cities):
            total_pop = self.city_populations[city_id]
            self.city_states[city_id] = {
                "susceptible": total_pop,
                "infected": 0,
                "recovered": 0,
                "vaccinated": 0
            }
        self.current_observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        self.total_infections = 0

    def _calculate_budget(self):
        city_densities = [city["density"] for city in self.city_manager.city_data.values()]
        total_density = sum(city_densities)
        max_density = max(city_densities)
        self.budget = self.BASE_BUDGET_PER_CITY * self.n_cities * (total_density / max_density)

    def get_infection_rate(self, city_id, base_rate=None):
        """
        Generate an *adjusted* infection rate for a city,
        factoring in:
        - The base rate (mean beta)
        - The fraction of the city that's vaccinated
        - The marketing/cleanliness factor
        - (Optionally) random variation (e.g., Poisson)
        """
        if base_rate is None:
            base_rate = self.MEAN_BETA

        # Fraction vaccinated in this city
        vaccinated_fraction = 0.0
        if self.city_populations[city_id] > 0:
            vaccinated_fraction = (
                self.city_states[city_id]["vaccinated"] / self.city_populations[city_id]
            )

        # Marketing/cleanliness factor for this city (assumed in [0,1])
        # The higher this factor, the more it reduces infection rate.
        marketing_factor = self.marketing_cleanliness_factors.get(city_id, 0.0)
        
        # Simple model: reduce infection rate by some proportion of vaccinated
        # plus some proportion from marketing. 
        # Feel free to tweak the formula for your needs.
        # For example:
        #   - vaccination can reduce the base rate up to 70% if everyone is vaccinated
        #   - marketing can reduce up to 30% if marketing_factor=1
        effective_rate = base_rate
        effective_rate *= (1.0 - 0.7 * vaccinated_fraction)  # e.g. 70% max reduction
        effective_rate *= (1.0 - 0.3 * marketing_factor)     # e.g. 30% max reduction

        # Then you might still want random variation around that effective rate:
        poisson_lambda = effective_rate * 10
        final_rate = np.random.poisson(poisson_lambda) / 10

        # Or skip Poisson if you prefer a direct approach
        # final_rate = effective_rate

        return min(final_rate, 1.0)
    

    def calculate_transmission_probability(self, source_city, target_city):
        """Calculate probability of disease transmission between cities."""
        source_infection_percentage = (
            self.city_states[source_city]["infected"] / 
            self.city_populations[source_city]
        )
        

        # If infection percentage is below threshold, no external transmission
        if source_infection_percentage < self.TRANSMISSION_THRESHOLD:
            return 0.0
            
        edge_weight = self.graph[source_city][target_city]["weight"]
        base_probability = 1 / (1 + edge_weight/100)  # Normalize distance effect
        
        return base_probability * source_infection_percentage

    def update_infections(self):
        """Update infection states for all cities."""
        new_infections = {}
        
        # Calculate new infections from city-to-city transmission
        for city_id in range(self.n_cities):
            neighbors = list(self.graph.neighbors(city_id))
            total_new_infected = 0
            
            # Internal spread
            current_rate = self.get_infection_rate()
            internal_new_cases = int(
                self.city_states[city_id]["susceptible"] * 
                current_rate * 
                (self.city_states[city_id]["infected"] / self.city_populations[city_id])
            )
            total_new_infected += internal_new_cases
            
            # External spread from neighbors
            for neighbor in neighbors:
                if self.calculate_transmission_probability(neighbor, city_id) > np.random.random():
                    transmission_rate = self.get_infection_rate(self.MEAN_BETA * 0.5)  # Reduced rate for external transmission
                    external_new_cases = int(
                        self.city_states[city_id]["susceptible"] * 
                        transmission_rate
                    )
                    total_new_infected += external_new_cases
            
            # Recovery
            recovered = int(self.city_states[city_id]["infected"] * self.GAMMA)
            
            new_infections[city_id] = {
                "new_infected": min(total_new_infected, self.city_states[city_id]["susceptible"]),
                "recovered": recovered
            }
        
        # Apply updates
        for city_id, updates in new_infections.items():
            self.city_states[city_id]["infected"] += updates["new_infected"]
            self.city_states[city_id]["susceptible"] -= updates["new_infected"]
            self.city_states[city_id]["infected"] -= updates["recovered"]
            self.city_states[city_id]["recovered"] += updates["recovered"]

    def calculate_lockdown_cost(self, city_id):
        city_density = self.city_manager.get_city_data(city_id)["density"]
        max_density = max(city["density"] for city in self.city_manager.city_data.values())
        density_factor = city_density / max_density
        return self.BASE_LOCKDOWN_COST * density_factor

    def calculate_vaccination_cost(self, city_id):
        city_density = self.city_manager.get_city_data(city_id)["density"]
        max_density = max(city["density"] for city in self.city_manager.city_data.values())
        density_factor = city_density / max_density
        return self.BASE_VACCINATION_COST * (1 - density_factor)
    
    def step():
        """Execute one time step within the environment.
        
        Args:
            action: 
        
        Returns:
            observation: Current state of the environment
            cost: Reward for the action taken
            done: Whether the episode has ended
            info: Additional information
        """

if __name__ == "__main__":
    env = TravelEnv()
    env.city_states[0]["infected"] = 100  # Start with 100 infected in first city
    print("Initial state:", env.city_states)
    env.update_infections()
    print("After update:", env.city_states)