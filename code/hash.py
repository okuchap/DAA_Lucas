import numpy as np
import matplotlib.pyplot as plt


# The estimated value
alpha = 23.5944
beta = 9.0222

@np.vectorize
# should be care about overflow, underflow
def sigmoid(x):
    sigmoid_range = 34.538776394910684

    if x <= -sigmoid_range:
        return 1e-15
    if x >= sigmoid_range:
        return 1.0 - 1e-15

    return 1.0 / (1.0 + np.exp(-x))


@np.vectorize
# hash_ubd=55, hash_slope=3, hash_center=1.5
def hash(exp_reward, center=1.5, slope=3, ubd=55):
    return ubd * sigmoid(slope * (exp_reward - center))


# plot
def plot_hash(center=1.5, slope=3, ubd=55):
    x = np.arange(0,3.5,0.02)
    y = hash(exp_reward=x, center=center, slope=slope, ubd=ubd)
    plt.plot(x,y, label='sigmoid')
    z = alpha + x * beta
    plt.plot(x,z, label='MLE')
    plt.title('Hash supply(EH/s), center={}'.format(center))
    plt.legend()
    plt.xlabel('Expected Reward (USD/EH)')
    plt.ylabel('Hash supply (EH/s)')
    testx = [1.578, 3.12, 3.12, 1.578]
    testy = [0, 0, ubd, ubd]
    plt.fill(testx,testy,color="y",alpha=0.3)
    plt.show()

    return None