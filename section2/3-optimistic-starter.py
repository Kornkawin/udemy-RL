import numpy as np
import matplotlib.pyplot as plt

NUM_TRIALS = 10000
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]


class BanditArm:
    def __init__(self, p):
        self.p = p
        # optimistic initial value
        # set to 5 to encourage exploration
        self.p_estimate = 5.
        # init N to 1
        # to keep the effect of optimistic initial value
        # in the first update
        self.N = 1

    def pull(self):
        # IN Bernoulli Distribution
        return np.random.random() < self.p

    def update(self, reward):
        self.N += 1
        self.p_estimate = ((self.N - 1) * self.p_estimate + reward) / self.N


def experiment():
    # initialize bandits
    bandits = [BanditArm(p) for p in BANDIT_PROBABILITIES]

    # to store the rewards
    rewards = np.zeros(NUM_TRIALS)

    for i in range(NUM_TRIALS):
        j = np.argmax([b.p_estimate for b in bandits])
        x = bandits[j].pull()
        rewards[i] = x
        bandits[j].update(x)

    for idx, b in enumerate(bandits):
        print(f"mean estimate of Bandit#{idx}:", b.p_estimate)
    print("total reward earned:", rewards.sum())
    print("overall win rate:", rewards.sum() / NUM_TRIALS)
    print("num times selected each bandit:", [b.N-1 for b in bandits])

    # plot the results
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
    plt.ylim([-0.1, 1.1])
    # plt.xscale('log')
    plt.plot(win_rates, label='win_rates')
    plt.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_PROBABILITIES), label='optimal')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    experiment()
