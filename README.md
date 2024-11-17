# Multi-Armed Bandit Algorithm Implementation

This project provides an implementation of two common algorithms for solving the multi-armed bandit problem: **Epsilon Greedy** and **Thompson Sampling**. The implementation includes features for experimentation, visualization of results, and saving the output to CSV files.

## Features

- **Bandit Algorithms**:
  - Epsilon Greedy: Explores and exploits bandit arms based on an epsilon decay strategy.
  - Thompson Sampling: Uses Bayesian updating to select the most promising bandit arm.

- **Visualization**:
  - Average Reward (Linear and Log Scale)
  - Cumulative Regret (Linear and Log Scale)
  - Combined plots for clear comparison

- **CSV Output**:
  - Stores results for both algorithms in separate CSV files with columns:
    - `Bandit`: The bandit index.
    - `Reward`: The reward received.
    - `Algorithm`: The algorithm used.

## Requirements

Ensure you have the following libraries installed:

- `numpy`
- `matplotlib`
- `logging`
- `csv`

You can install these using pip:

```bash
pip install numpy matplotlib
```

## How to Run

1. Save the script as a `.py` file (e.g., `bandit.py`).
2. Run the script:

```bash
python bandit.py
```

The script will:
- Run experiments using Epsilon Greedy and Thompson Sampling.
- Save results to `epsilon_greedy_results.csv` and `thompson_sampling_results.csv`.
- Generate plots for visualization.

## Outputs

1. **CSV Files**:
   - `epsilon_greedy_results.csv`
   - `thompson_sampling_results.csv`

   Each file includes:
   - `Bandit`: Index of the bandit pulled.
   - `Reward`: Reward obtained for the pull.
   - `Algorithm`: Name of the algorithm.

2. **Plots**:
   - Combined plots for each algorithm:
     - Average Reward (Linear and Log Scale)
     - Cumulative Regret (Linear and Log Scale)
   - Comparison plot for Average Reward between Epsilon Greedy and Thompson Sampling.

## Key Functions

### Epsilon Greedy
- **experiment**: Simulates the Epsilon Greedy algorithm for a given number of trials.
  - Initializes bandits with true reward probabilities.
  - Pulls arms based on epsilon decay strategy, balancing exploration and exploitation.

- **report**: Runs the experiment, saves results to a CSV file, and generates combined plots showing average rewards and cumulative regrets.

### Thompson Sampling
- **experiment**: Simulates the Thompson Sampling algorithm for a given number of trials.
  - Employs Bayesian updating to sample from posterior distributions of bandit rewards.

- **report**: Runs the experiment, saves results to a CSV file, and generates combined plots showing average rewards and cumulative regrets.

### Visualization
- **plot_combined**: Generates a 2x2 grid plot for:
  - Average Reward (Linear and Log Scale)
  - Cumulative Regret (Linear and Log Scale)

### Comparison
- **comparison**: Compares the average reward between Epsilon Greedy and Thompson Sampling in a single plot.
  - Highlights the relative performance of each algorithm in selecting optimal bandits.

## Example

The script uses the following configuration for testing:

```python
BANDIT_REWARDS = [1.0, 2.0, 3.0, 4.0]
NUM_TRIALS = 20000
```

You can modify `BANDIT_REWARDS` and `NUM_TRIALS` to test with different configurations.

## License

This project is provided for educational purposes. Feel free to use and modify it as needed.

