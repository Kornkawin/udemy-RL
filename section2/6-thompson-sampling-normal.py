import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(1)
NUM_TRIALS = 2000
BANDIT_MEANS = [1, 2, 3]


class BanditArm:
    def __init__(self, true_mean):
        self.true_mean = true_mean
        # parameters for mu (prior) is N(0,1)
        self.predicted_mean = 0
        self.lambda_ = 1
        self.sum_x = 0  # for convenience
        self.tau = 1
        self.N = 0

    def pull(self):
        std = np.sqrt(1. / self.tau)
        return self.true_mean + (std * np.random.randn())

    def sample(self):
        std_ = np.sqrt(1. / self.lambda_)
        return self.predicted_mean + (std_ * np.random.randn())

    def update(self, reward):
        self.lambda_ += self.tau
        self.sum_x += reward
        self.predicted_mean = self.tau * self.sum_x / self.lambda_
        self.N += 1


def plot(bandits, trial):
    x = np.linspace(-3, 6, 200)
    for b in bandits:
        y = norm.pdf(x, b.predicted_mean, np.sqrt(1/b.lambda_))
        plt.plot(x, y, label=f"real_mean={b.true_mean:.4f}, num_plays={b.N}")
    plt.title(f"Bandit distributions after {trial} trials")
    plt.legend()
    plt.show()


def experiment():
    bandits = [BanditArm(mean) for mean in BANDIT_MEANS]
    sample_points = [5, 10, 20, 50, 100, 200, 500, 1000, 1500, 1999]
    rewards = np.zeros(NUM_TRIALS)
    for i in range(NUM_TRIALS):
        j = np.argmax([b.sample() for b in bandits])
        if i in sample_points:
            plot(bandits, i)
        x = bandits[j].pull()
        rewards[i] = x
        bandits[j].update(x)
    print("total reward earned:", rewards.sum())
    print("average reward earned:", rewards.sum() / NUM_TRIALS)
    print("num times selected each bandit:", [b.N for b in bandits])


if __name__ == "__main__":
    experiment()
