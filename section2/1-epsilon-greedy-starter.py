import matplotlib.pyplot as plt
import numpy as np

# Number of trials
NUM_TRIALS = 10000
# Epsilon value for epsilon-greedy algorithm
epsilon = 0.1
# True win probability of each bandit arm
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]


class BanditArm:
    def __init__(self, win_rate):
        self.win_rate = win_rate
        self.num_pulled = 0
        self.estimated_win_rate = 0

    def pull(self):
        # IN Bernoulli Distribution
        # reward x is {0, 1}
        # with 1 being a win
        # and 0 being a loss
        # x~Bernoulli(TRUE_WIN_RATE)
        return np.random.random() < self.win_rate

    def update(self, reward):
        self.num_pulled += 1
        self.estimated_win_rate += (reward - self.estimated_win_rate) / self.num_pulled


def experiment():
    # initialize bandits
    bandits = [BanditArm(p) for p in BANDIT_PROBABILITIES]
    rewards = np.zeros(NUM_TRIALS)
    num_explored = 0
    num_exploited = 0
    num_optimal = 0

    # TRUE OPTIMAL BANDIT
    optimal_j = np.argmax([b.win_rate for b in bandits])
    print("optimal j:", optimal_j)

    for i in range(NUM_TRIALS):
        # use epsilon-greedy to select the next bandit
        if np.random.random() < epsilon:
            # randomly explore
            num_explored += 1
            j = np.random.randint(len(bandits))
        else:
            # exploit best bandit
            num_exploited += 1
            j = np.argmax([b.estimated_win_rate for b in bandits])

        # count how many times we selected the optimal bandit
        if j == optimal_j:
            num_optimal += 1

        # pull the arm for the bandit with the largest sample win_rate
        x = bandits[j].pull()
        # update rewards log
        rewards[i] = x
        # update the distribution for the bandit whose arm we just pulled
        bandits[j].update(x)

    for idx, b in enumerate(bandits):
        print(f"win_rate estimate of bandit #{idx}:", b.estimated_win_rate)
    print("total reward earned:", rewards.sum())
    print("overall win rate:", rewards.sum() / NUM_TRIALS)
    print("num_times_explored:", num_explored)
    print("num_times_exploited:", num_exploited)
    print("num times selected optimal bandit:", num_optimal)

    # plot the results
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
    plt.plot(win_rates)
    plt.plot(np.ones(NUM_TRIALS) * np.max(BANDIT_PROBABILITIES))
    plt.show()


if __name__ == "__main__":
    experiment()
