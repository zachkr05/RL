class CityManager:
    def __init__(self):
        self.city_data = self.initialize_city_data()
        self.n_cities = len(self.city_data)

    @staticmethod
    def initialize_city_data():
        """
        Initialize city data with population, area, and calculate population density.
        """
        city_data = {
            0: {"name": "NYC", "pop": 8400000, "area": 302.6},  # Area in square miles
            1: {"name": "Buffalo", "pop": 278000, "area": 52.5},
            2: {"name": "Rochester", "pop": 211000, "area": 37.1},
            3: {"name": "Syracuse", "pop": 148000, "area": 25.6},
            4: {"name": "Albany", "pop": 99000, "area": 21.4}
        }

        # Calculate population density for each city
        for city_id, city in city_data.items():
            city["density"] = city["pop"] / city["area"]  # Population density (pop/sq mile)
        
        return city_data

    def get_city_data(self, city_id):
        """
        Retrieve data for a specific city by its ID.
        """
        return self.city_data.get(city_id, None)

    def set_city_data(self, city_id, name=None, pop=None, area=None):
        """
        Update data for a specific city by its ID.
        """
        if city_id not in self.city_data:
            raise ValueError(f"City ID {city_id} does not exist.")

        if name is not None:
            self.city_data[city_id]["name"] = name
        if pop is not None:
            self.city_data[city_id]["pop"] = pop
        if area is not None:
            self.city_data[city_id]["area"] = area

        # Recalculate density
        self.city_data[city_id]["density"] = self.city_data[city_id]["pop"] / self.city_data[city_id]["area"]

    def add_city(self, city_id, name, pop, area):
        """
        Add a new city to the data.
        """
        if city_id in self.city_data:
            raise ValueError(f"City ID {city_id} already exists.")
        self.city_data[city_id] = {"name": name, "pop": pop, "area": area}
        self.city_data[city_id]["density"] = pop / area
        self.n_cities += 1

    def remove_city(self, city_id):
        """
        Remove a city from the data.
        """
        if city_id in self.city_data:
            del self.city_data[city_id]
            self.n_cities -= 1
        else:
            raise ValueError(f"City ID {city_id} does not exist.")

    def total_population(self):
        """
        Calculate the total population across all cities.
        """
        return sum(city["pop"] for city in self.city_data.values())

    def total_area(self):
        """
        Calculate the total area across all cities.
        """
        return sum(city["area"] for city in self.city_data.values())

    def average_density(self):
        """
        Calculate the average population density across all cities.
        """
        return sum(city["density"] for city in self.city_data.values()) / self.n_cities

    def city_names(self):
        """
        Get a list of all city names.
        """
        return [city["name"] for city in self.city_data.values()]

    def get_city_population(self, city_id):
        """
        Get the population of a specific city.
        """
        city = self.get_city_data(city_id)
        return city["pop"] if city else None

    def get_city_density(self, city_id):
        """
        Get the population density of a specific city.
        """
        city = self.get_city_data(city_id)
        return city["density"] if city else None

    def update_city_population(self, city_id, new_population):
        """
        Update the population of a specific city and recalculate its density.
        """
        if city_id not in self.city_data:
            raise ValueError(f"City ID {city_id} does not exist.")
        self.city_data[city_id]["pop"] = new_population
        self.city_data[city_id]["density"] = new_population / self.city_data[city_id]["area"]

    def update_city_area(self, city_id, new_area):
        """
        Update the area of a specific city and recalculate its density.
        """
        if city_id not in self.city_data:
            raise ValueError(f"City ID {city_id} does not exist.")
        self.city_data[city_id]["area"] = new_area
        self.city_data[city_id]["density"] = self.city_data[city_id]["pop"] / new_area

    def get_density(self, city_id):
        """
        Get the population density of a specific city.
        """
        city = self.get_city_data(city_id)
        if city:
            return city["density"]
        else:
            raise ValueError(f"City ID {city_id} does not exist.")