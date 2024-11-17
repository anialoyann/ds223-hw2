from abc import ABC, abstractmethod
import logging
import numpy as np
import matplotlib.pyplot as plt
import csv

# Logger setup
logging.basicConfig()
logger = logging.getLogger("MAB Application")
logger.setLevel(logging.DEBUG)

class Bandit(ABC):
    """
    Abstract base class for bandit algorithms.
    """

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @classmethod
    @abstractmethod
    def experiment(cls):
        pass

    @classmethod
    @abstractmethod
    def report(cls):
        pass

class Visualization():
    """
    Visualization methods for plotting results of bandit experiments.
    """

    @staticmethod
    def plot_combined(rewards, regrets, num_trials, optimal_bandit_reward, method_name):
        cumulative_rewards = np.cumsum(rewards)
        average_reward = cumulative_rewards / (np.arange(num_trials) + 1)
        cumulative_regrets = optimal_bandit_reward * np.arange(1, num_trials + 1) - cumulative_rewards

        fig, ax = plt.subplots(2, 2, figsize=(12, 10))

        ax[0, 0].plot(average_reward, label="Average Reward")
        ax[0, 0].axhline(optimal_bandit_reward, color="g", linestyle="--", label="Optimal Bandit Reward")
        ax[0, 0].legend()
        ax[0, 0].set_title(f"{method_name}: Average Reward (Linear Scale)")
        ax[0, 0].set_xlabel("Number of Trials")
        ax[0, 0].set_ylabel("Average Reward")

        ax[0, 1].plot(average_reward, label="Average Reward")
        ax[0, 1].axhline(optimal_bandit_reward, color="g", linestyle="--", label="Optimal Bandit Reward")
        ax[0, 1].legend()
        ax[0, 1].set_title(f"{method_name}: Average Reward (Log Scale)")
        ax[0, 1].set_xlabel("Number of Trials")
        ax[0, 1].set_ylabel("Average Reward")
        ax[0, 1].set_yscale("log")

        ax[1, 0].plot(cumulative_regrets, label="Cumulative Regret")
        ax[1, 0].legend()
        ax[1, 0].set_title(f"{method_name}: Cumulative Regret (Linear Scale)")
        ax[1, 0].set_xlabel("Number of Trials")
        ax[1, 0].set_ylabel("Cumulative Regret")

        ax[1, 1].plot(cumulative_regrets, label="Cumulative Regret")
        ax[1, 1].legend()
        ax[1, 1].set_title(f"{method_name}: Cumulative Regret (Log Scale)")
        ax[1, 1].set_xlabel("Number of Trials")
        ax[1, 1].set_ylabel("Cumulative Regret")
        ax[1, 1].set_yscale("log")

        plt.tight_layout()
        plt.show()

class EpsilonGreedy(Bandit):
    def __init__(self, p):
        self.p = p
        self.p_estimate = 0
        self.N = 0

    def pull(self):
        return np.random.randn() + self.p

    def update(self, x):
        self.N += 1
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N

    def __repr__(self):
        return f"EpsilonGreedy Arm with true mean {self.p:.4f}"

    @classmethod
    def experiment(cls, bandit_probabilities, num_trials, initial_epsilon=0.1, min_epsilon=0.02):
        bandits = [cls(p) for p in bandit_probabilities]
        rewards = []
        optimal_bandit = np.argmax([b.p for b in bandits])

        for i in range(1, num_trials + 1):
            epsilon = max(initial_epsilon / i, min_epsilon)
            if np.random.random() < epsilon:
                chosen_bandit = np.random.randint(len(bandits))
            else:
                chosen_bandit = np.argmax([b.p_estimate for b in bandits])

            reward = bandits[chosen_bandit].pull()
            rewards.append(reward)
            bandits[chosen_bandit].update(reward)

        return bandits, rewards

    @classmethod
    def report(cls, bandit_probabilities, num_trials):
        bandits, rewards = cls.experiment(bandit_probabilities, num_trials)
        optimal_bandit_reward = max(bandit_probabilities)

        with open("epsilon_greedy_results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Bandit", "Reward", "Algorithm"])
            for i, reward in enumerate(rewards):
                writer.writerow([i % len(bandits), reward, "Epsilon Greedy"])

        logger.info("Epsilon Greedy Results:")
        for i, b in enumerate(bandits):
            logger.info(f"Bandit {i}: True Mean = {b.p:.4f}, Estimated Mean = {b.p_estimate:.4f}, Pulled = {b.N}")

        Visualization.plot_combined(rewards, rewards, num_trials, optimal_bandit_reward, "Epsilon Greedy")

class ThompsonSampling(Bandit):
    def __init__(self, m):
        self.m = m
        self.m_estimate = 0
        self.lambda_ = 1
        self.tau = 1
        self.sum_x = 0
        self.N = 0

    def pull(self):
        return np.random.randn() / np.sqrt(self.tau) + self.m

    def sample(self):
        return np.random.randn() / np.sqrt(self.lambda_) + self.m_estimate

    def update(self, x):
        self.N += 1
        self.sum_x += x
        self.lambda_ += self.tau
        self.m_estimate = self.sum_x / self.lambda_

    def __repr__(self):
        return f"ThompsonSampling Arm with true mean {self.m:.4f}"

    @classmethod
    def experiment(cls, bandit_probabilities, num_trials):
        bandits = [cls(p) for p in bandit_probabilities]
        rewards = []

        for _ in range(num_trials):
            chosen_bandit = np.argmax([b.sample() for b in bandits])
            reward = bandits[chosen_bandit].pull()
            rewards.append(reward)
            bandits[chosen_bandit].update(reward)

        return bandits, rewards

    @classmethod
    def report(cls, bandit_probabilities, num_trials):
        bandits, rewards = cls.experiment(bandit_probabilities, num_trials)
        optimal_bandit_reward = max(bandit_probabilities)

        with open("thompson_sampling_results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Bandit", "Reward", "Algorithm"])
            for i, reward in enumerate(rewards):
                writer.writerow([i % len(bandits), reward, "Thompson Sampling"])

        logger.info("Thompson Sampling Results:")
        for i, b in enumerate(bandits):
            logger.info(f"Bandit {i}: True Mean = {b.m:.4f}, Estimated Mean = {b.m_estimate:.4f}, Pulled = {b.N}")

        Visualization.plot_combined(rewards, rewards, num_trials, optimal_bandit_reward, "Thompson Sampling")

def comparison(bandit_rewards, num_trials):
    eg_bandits, eg_rewards = EpsilonGreedy.experiment(bandit_rewards, num_trials)
    ts_bandits, ts_rewards = ThompsonSampling.experiment(bandit_rewards, num_trials)

    optimal_bandit_reward = max(bandit_rewards)
    eg_cumulative_rewards = np.cumsum(eg_rewards) / (np.arange(1, num_trials + 1))
    ts_cumulative_rewards = np.cumsum(ts_rewards) / (np.arange(1, num_trials + 1))

    plt.plot(eg_cumulative_rewards, label="Epsilon Greedy")
    plt.plot(ts_cumulative_rewards, label="Thompson Sampling")
    plt.axhline(optimal_bandit_reward, color="g", linestyle="--", label="Optimal Bandit Reward")
    plt.legend()
    plt.title("Comparison of Average Reward")
    plt.xlabel("Number of Trials")
    plt.ylabel("Average Reward")
    plt.show()

if __name__ == "__main__":
    BANDIT_REWARDS = [1.0, 2.0, 3.0, 4.0]
    NUM_TRIALS = 20000

    EpsilonGreedy.report(BANDIT_REWARDS, NUM_TRIALS)
    ThompsonSampling.report(BANDIT_REWARDS, NUM_TRIALS)
    comparison(BANDIT_REWARDS, NUM_TRIALS)
