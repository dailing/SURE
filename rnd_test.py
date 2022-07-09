# %%
from cv2 import mean
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

# %%
sample = np.random.rand(100)

def score1(x):
    result = [np.sum(np.abs(x-x[i])) for i in range(len(x))]
    return sum(result) / len(result)

def scoreL1(x):
    meanx = np.mean(x)
    return np.sum(np.abs(x-meanx))


# %%
samples = []
samples2 = []
for i in range(5, 100):
    for _ in range(10):
        sample = np.random.rand(i)
        samples.append((
            score1(sample),
            scoreL1(sample))
        )

# %%
samples = np.array(samples)

# %%
plt.figure(figsize=(10, 10))
plt.scatter(samples[:,0], samples[:,1], s=1)
max_val = np.max(samples)
plt.plot([0, max_val], [0, max_val], 'r')

# %% [markdown]
sample = np.random.rand(30)-0.5
x = np.linspace(-1, 1, 100)
y = np.array([np.mean(np.abs((sample-i)**1)) for i in x])
plt.plot(x, y)
plt.plot([np.mean(sample), np.mean(sample)], [0,1])
print(np.mean(sample) - x[np.argmin(y)])


# %% simulate sign(F)*f VS E(f) ^2

epision = 0.0
sample1 = []
sample2 = []
var = []
for i in range(10, 1000, 10):
    for _ in range(100):
        ratio = np.random.rand() * 0.0
        sigma =  0.8
        F = np.random.rand(i) - 0.5 + ratio
        f = np.random.normal(0,sigma * np.random.rand(),i) + F
        var = np.var(F-f)
        # print(var)
        sign_F = np.sign(F)
        score1 = sign_F*(epision-f)
        score1[score1 < 0] = 0

        score2 = abs(np.mean((F-f)))**2
        sample1.append(np.mean(score1))
        sample2.append(score2)
sample1 = np.array(sample1)
sample2 = np.array(sample2)
# sample1 -= np.mean(sample1)
# sample2 -= np.mean(sample2)
# sample1 /= np.std(sample1)
# sample2 /= np.std(sample2)

# %%
# plt.scatter(sample1, sample2, s=0.1,alpha=0.5)
# g = sns.violinplot(sample1, sample2)
# plt.show()

# X = (sample1 + sample2)/2
# Y = sample1-sample2
# plt.scatter((sample1+sample1)/2, sample1-sample2 , s=0.1,alpha=0.5)
# plt.show()

# reg = LinearRegression().fit(X, y)


# %%
