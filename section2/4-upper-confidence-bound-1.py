import numpy as np
import matplotlib.pyplot as plt

NUM_TRIALS = 100000
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]

class BanditArm:
    def __init__(self, p):
        # p is win rate
        self.p = p
        self.p_estimate = 0.
        self.N = 0

    def pull(self):
        return np.random.random() < self.p

    def update(self, reward):
        self.N += 1
        self.p_estimate = ((self.N - 1) * self.p_estimate + reward) / self.N


def ucb1(mean, n, n_j):
    return mean + np.sqrt(2 * np.log(n) / n_j)


def experiment():
    bandits = [BanditArm(p) for p in BANDIT_PROBABILITIES]
    rewards = np.empty(NUM_TRIALS)
    total_plays = 0

    # initialization: play each bandit once
    # to avoid division by zero
    for j in range(len(bandits)):
        x = bandits[j].pull()
        total_plays += 1
        bandits[j].update(x)

    for i in range(NUM_TRIALS):
        j = np.argmax([ucb1(b.p_estimate, total_plays, b.N) for b in bandits])
        x = bandits[j].pull()
        total_plays += 1
        bandits[j].update(x)
        # collect for plot
        rewards[i] = x
    cum_avg = np.cumsum(rewards) / (np.arange(NUM_TRIALS) + 1)

    for idx, b in enumerate(bandits):
        print(f"mean estimate of Bandit#{idx}:", b.p_estimate)
    print("total reward earned:", rewards.sum())
    print("overall win rate:", rewards.sum() / NUM_TRIALS)
    print("num times selected each bandit:", [b.N-1 for b in bandits])

    plt.plot(cum_avg, label='cumulative average reward')
    plt.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_PROBABILITIES), label='optimal')
    plt.xscale('log')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    experiment()
