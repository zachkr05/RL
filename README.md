# AI-Driven Pandemic Outbreak Simulation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenAI Gym](https://img.shields.io/badge/OpenAI%20Gym-Compatible-brightgreen)](https://gym.openai.com/)

An AI-powered simulation environment for optimizing pandemic response strategies using Deep Reinforcement Learning (DQN). Models disease spread across interconnected cities with dynamic containment measures.

![Simulation Overview](assets/simulation-demo.gif) <!-- Replace with actual path -->

## Table of Contents
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Key Features

### 🏙️ City Network Modeling
- Population density dynamics across 5+ cities
- Travel route management with NetworkX graphs
- Infection rate calculations based on:
  - Population density
  - Vaccination coverage
  - Public health initiatives
- Susceptible-Infected-Recovered (SIR) model integration

### 🤖 AI Response Management
- DQN agent implementation with Stable-Baselines3
- Action space includes:
  - Travel restriction policies
  - Vaccination campaign prioritization
  - Public health marketing investments
- Reward function balancing:
  - Economic costs (lockdowns/vaccinations)
  - Infection control outcomes
  - Healthcare system capacity

### 📊 Advanced Simulation
- Internal/external transmission modeling
- Dynamic budget allocation system
- Real-time strategy adaptation
- MATLAB/Python data analysis integration
- Infection trajectory visualization

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
```bash
# Clone repository
git clone https://github.com/zachkr05/pandemic-rl-simulation.git
cd pandemic-rl-simulation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
