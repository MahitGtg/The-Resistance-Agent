# The Resistance Game Agent
![images](https://github.com/user-attachments/assets/99e8d021-26d6-44c3-bc32-0fd9a69696a4)

An intelligent agent for playing The Resistance, implemented using enhanced probabilistic reasoning with Q-Learning.(MyAgent)

## Getting Started

### 1. Download the Project

To download the project, use the following wget command:

```bash
wget https://github.com/MahitGtg/The-Resistance-Agent/tree/main/the_resistance
```

### 2. Extract the Files

If the downloaded file is a zip archive, extract it:

```bash
unzip [downloaded_file_name].zip
```

### 3. Navigate to the Project Directory

```bash
cd the_resistance
```

## Running the Game

To run a single game:

```bash
python run_game.py
```

To run a tournament:

```bash
python run_tournament.py
```

## Comparing Agents

The project includes four different agents:

1. `myagent.py`: My implemented agent
2. `basic_agent.py`: A basic implementation
3. `random_agent.py`: An agent that makes random decisions
4. `satisfactory_agent.py`: A more advanced implementation

You can compare the performance of your agent against these other agents by running the tournament and analyzing the results.

## Features of MyAgent

- Dynamic spy probability updates
- Trust score calculation for each player
- Q-Learning for action selection
- Integration of expert player strategies

## Performance

In sample tests against reference agents:
- vs RandomAgent: 78% win rate
- vs BasicAgent: 65% win rate
- vs SatisfactoryAgent: 53% win rate
