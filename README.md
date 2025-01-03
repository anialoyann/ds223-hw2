# Multi-Armed Bandit Algorithms: Epsilon-Greedy and Thompson Sampling

## Overview
This project implements and compares two popular algorithms for solving the Multi-Armed Bandit (MAB) problem:

1. **Epsilon-Greedy**: Balances exploration and exploitation using a decaying epsilon parameter.
2. **Thompson Sampling**: A Bayesian approach that uses posterior sampling to guide decision-making.

The project includes visualizations and logging to help understand how these algorithms perform over time.

---

## Features
### Implemented Algorithms
- **Epsilon-Greedy**:
  - Adjusts exploration rate dynamically as trials progress.
  - Stores estimates of reward probabilities for each bandit.

- **Thompson Sampling**:
  - Samples from posterior distributions to choose the optimal bandit.
  - Updates posterior parameters based on observed rewards.

### Visualization
The `Visualization` class provides:
1. Combined plots for cumulative rewards and cumulative regrets (linear and log scales).
2. Posterior distribution plots for Thompson Sampling at specific trial milestones.

### Comparison
The project includes a function to compare the performance of Epsilon-Greedy and Thompson Sampling by plotting average rewards over trials.

---

## File Structure
- `main.py`: Main script to run the experiments and generate reports.
- `epsilon_greedy_results.csv`: Results from the Epsilon-Greedy experiment.
- `thompson_sampling_results.csv`: Results from the Thompson Sampling experiment.
- `plots/`: Directory for visualizations generated during the experiments.

---

## Usage
### Prerequisites
Ensure the following Python libraries are installed:
- `numpy`
- `matplotlib`
- `scipy`
- `logging`

Install them using:
```bash
pip install numpy matplotlib scipy
```

### Running the Code
To run the experiments and generate reports:
```bash
python main.py
```

### Expected Output
1. **CSV Files**: Results of the experiments are saved as `epsilon_greedy_results.csv` and `thompson_sampling_results.csv`.
2. **Plots**: Visual comparisons of cumulative rewards, regrets, and posterior distributions.
3. **Logs**: Detailed performance of each bandit is logged in the console.

---

## Key Classes and Methods
### `Bandit`
Abstract base class for implementing bandit algorithms.
- `experiment(cls, bandit_probabilities, num_trials)`: Runs the experiment.
- `report(cls, bandit_probabilities, num_trials)`: Generates and saves reports.

### `Visualization`
Provides static methods for plotting results:
- `plot_combined(rewards, regrets, num_trials, optimal_bandit_reward, method_name)`: Visualizes cumulative rewards and regrets.
- `plot_bandit_distributions(bandits, trial)`: Plots posterior distributions for Thompson Sampling.

### `EpsilonGreedy`
Implements the Epsilon-Greedy algorithm.

### `ThompsonSampling`
Implements the Thompson Sampling algorithm.

---

## Sample Experiment
### Parameters
- Bandit reward probabilities: `[1.0, 2.0, 3.0, 4.0]`
- Number of trials: `20,000`

### Results
1. Epsilon-Greedy dynamically balances exploration and exploitation, converging to the optimal bandit over time.
2. Thompson Sampling leverages Bayesian inference, showing faster convergence in many cases.

