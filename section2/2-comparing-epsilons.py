import numpy as np
import matplotlib.pyplot as plt


class BanditArm:
    def __init__(self, m):
        self.m = m
        self.num_pulled = 0
        self.estimated_m = 0

    def pull(self):
        # IN Normal Distribution x~N(m, 1)
        return np.random.randn() + self.m

    def update(self, reward):
        self.num_pulled += 1
        self.estimated_m += (reward - self.estimated_m) / self.num_pulled


def run_experiment(m1, m2, m3, eps, N):
    bandits = [BanditArm(m1), BanditArm(m2), BanditArm(m3)]
    means = np.array([m1, m2, m3])
    true_best_bandit = np.argmax(means)
    count_suboptimal = 0

    data = np.empty(N)

    for i in range(N):
        # Epsilon Greedy
        p = np.random.random()
        if p < eps:
            j = np.random.choice(len(bandits))
        else:
            j = np.argmax([b.estimated_m for b in bandits])

        x = bandits[j].pull()
        bandits[j].update(x)

        if j != true_best_bandit:
            count_suboptimal += 1

        # for the plot
        data[i] = x
    cum_avg = np.cumsum(data) / (np.arange(N) + 1)

    # plot moving average ctr
    plt.plot(cum_avg)
    plt.plot(np.ones(N) * m1)
    plt.plot(np.ones(N) * m2)
    plt.plot(np.ones(N) * m3)
    plt.xscale('log')
    plt.show()
    for b in bandits:
        print(b.estimated_m)
    print("percent suboptimal for epsilon = %s:" % eps, float(count_suboptimal) / N)
    return cum_avg


if __name__ == '__main__':
    m1 = 1.5
    m2 = 2.5
    m3 = 3.5
    N = 100000
    c_1 = run_experiment(m1, m2, m3, 0.1, N)
    c_05 = run_experiment(m1, m2, m3, 0.05, N)
    c_01 = run_experiment(m1, m2, m3, 0.01, N)

    # log scale plot
    plt.plot(c_1, label='eps = 0.1')
    plt.plot(c_05, label='eps = 0.05')
    plt.plot(c_01, label='eps = 0.01')
    plt.legend()
    plt.xscale('log')
    plt.show()

    # linear plot
    plt.plot(c_1, label='eps = 0.1')
    plt.plot(c_05, label='eps = 0.05')
    plt.plot(c_01, label='eps = 0.01')
    plt.legend()
    plt.show()

    print("Summary:")
    print("If we want to quicker converge to the optimal solution, use higher epsilon.")
    print("If we want a higher eventual rewards, use lower epsilon.")
