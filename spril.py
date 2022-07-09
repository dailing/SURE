# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def get_feature(theta):
    x = theta * np.cos(np.abs(theta))
    y = theta * np.sin(np.abs(theta))
    return x, y


theta = np.linspace(-15, 15, 1000)
x, y = get_feature(theta)


def F(x):
    th = np.arctan2(x[1], x[0])
    r = np.sqrt(x[0] ** 2 + x[1] ** 2)
    rr = round((r - th) / np.pi)
    # if rr % 2 == 0:
    th += rr * (np.pi)
    # else:
    # th -= rr * (np.pi )
    if rr % 2 == 1:
        th = -th
    return th


xyy = np.stack((x, y), axis=1)
th_reg = np.array([F(i) for i in xyy])
plt.scatter(theta, th_reg, s=1)

xx = (np.random.rand(3000, 2) - 0.5) * 30
rr = np.array([F(i) for i in xx])

plt.scatter(x, y, s=1, c=(theta > 0).astype(np.uint8), cmap='coolwarm')
plt.scatter(xx[:, 0], xx[:, 1], s=1, c=(rr > 0).astype(np.uint8), alpha=0.5, cmap='coolwarm')
plt.axis('equal')
plt.grid(True, which='both')

plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')

plt.show()

# %% make sample data
sample_r = (np.random.rand(2, 30000) - 0.5) * 30
sample_r[1, :] = sample_r[0, :] + (np.random.rand(1, 30000) - 0.5 + 4)
sample_r = sample_r.astype(np.float32)


# %%
class DeepSurvivalDS(Dataset):
    def __init__(self, sample_r):
        super().__init__()
        assert sample_r.shape[0] == 2
        d = sample_r[1] - sample_r[0]
        x1, y1 = get_feature(sample_r[0])
        x1 += np.random.random(x1.shape) * 0.2
        y1 += np.random.random(y1.shape) * 0.2
        x2, y2 = get_feature(sample_r[1])
        x2 += np.random.random(x2.shape) * 0.2
        y2 += np.random.random(y2.shape) * 0.2

        self.feat1 = np.stack((x1, y1), axis=1)
        self.feat2 = np.stack((x2, y2), axis=1)
        self.dist = d
        self.sign1 = np.sign(sample_r[0])
        self.sign2 = np.sign(sample_r[1])

    def __len__(self):
        return self.feat1.shape[0]

    def __getitem__(self, idx):
        return self.feat1[idx], self.feat2[idx], self.dist[idx], self.sign1[idx], self.sign2[idx]


class DeepSurModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        return x.squeeze()


# %% train test deep survival model
model = DeepSurModel()
episilon = 0.5
dataLoader = DataLoader(DeepSurvivalDS(sample_r), batch_size=500, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.9)
for epoch in range(1000):
    loss_accum = 0
    for feat1, feat2, dist, sign1, sign2 in dataLoader:
        res1 = model(feat1)
        res2 = model(feat2)
        L_cls = nn.functional.soft_margin_loss(res1, sign1) + nn.functional.soft_margin_loss(res2, sign2)
        L_reg = nn.functional.mse_loss(res2 - res1, dist) / 20
        loss = L_cls + L_reg * 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss)
        loss_accum += loss.item()
    scheduler.step()
    if epoch % 10 == 0:
        print(loss_accum)

print(res2, sign2)

# %%
output = model(torch.from_numpy(xyy).to(torch.float32)).detach().numpy()
plt.scatter(theta, output, s=1)
plt.xlabel('label')
plt.ylabel('predict')
plt.plot([-15, 15], [-15, 15], c='r')
plt.show()

# %%
x1_min, x1_max = -15, 15
x2_min, x2_max = -15, 15
resolution = 0.2
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                       np.arange(x2_min, x2_max, resolution))
t = torch.from_numpy(np.array([xx1.ravel(), xx2.ravel()]).T).to(torch.float32)
Z = model(t)
Z = Z.detach().numpy()
Z = Z.reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, alpha=0.8, cmap='coolwarm')
plt.colorbar()

plt.scatter(x, y, s=1, c=(theta > 0).astype(np.uint8), cmap='coolwarm')
# plt.scatter(xx[:,0], xx[:,1], s=1, c=(rr>0).astype(np.uint8), alpha=0.5, cmap='coolwarm')
plt.axis('equal')
plt.grid(True, which='both')

plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.show()

# %%
