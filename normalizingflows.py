import pandas as pd
import numpy as np
# import nflows
from nflows import transforms, distributions, flows
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from sklearn.preprocessing import StandardScaler
import random

x, y = datasets.make_circles(1000, noise=.1)
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.show(block=False)

num_layers = 5
base_dist = StandardNormal(shape=[2])

transforms = []
for _ in range(num_layers):
    transforms.append(ReversePermutation(features=2))
    transforms.append(MaskedAffineAutoregressiveTransform(features=2, 
                                                          hidden_features=4))
transform = CompositeTransform(transforms)

flow = Flow(transform, base_dist)
optimizer = optim.Adam(flow.parameters())

num_iter = 5000
for i in range(num_iter):
    x, y = datasets.make_moons(128, noise=.1)
    x = torch.tensor(x, dtype=torch.float32)
    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=x).mean()
    loss.backward()
    optimizer.step()
    
    if (i + 1) % 500 == 0:
        xline = torch.linspace(-1.5, 2.5, 100)
        yline = torch.linspace(-.75, 1.25, 100)
        xgrid, ygrid = torch.meshgrid(xline, yline)
        xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

        with torch.no_grad():
            zgrid = flow.log_prob(xyinput).exp().reshape(100, 100)

        plt.contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy())
        plt.title('iteration {}'.format(i + 1))
        plt.show(block=False)


#-------------------------
# Social Science DGP
#-------------------------



# Simulate the data
np.random.seed(42)
n = 1000

education = np.where(
    np.random.rand(n) < 0.5,
    np.random.normal(12, 1, n),
    np.random.normal(18, 1.5, n)
)
education = education.clip(8, 21)
income = np.random.exponential(scale=30, size=n) + 20  # most people under 80, a few over 150
income = income.clip(10, 200)

age = np.concatenate([
    np.random.normal(25, 3, n // 3),
    np.random.normal(65, 7, n - n // 3)
])
np.random.shuffle(age)
age = age.clip(18, 90)
gender = np.random.binomial(1, 0.5, size=n)


# Logistic model for voting
linpred = (
    -3.5
    + 0.6 * ((education - 16)**2 < 4).astype(float)  # sweet spot for voting likelihood
    + 0.02 * np.sqrt(income)                        # diminishing returns
    + 0.04 * age
    - 0.5 * (gender == 1) * (income > 80)           # weird gender-income interaction
)
prob_voted = 1 / (1 + np.exp(-linpred))
voted = np.random.binomial(1, prob_voted)


# Raw DataFrame
df = pd.DataFrame({
    "education": education,
    "income": income,
    "age": age,
    "gender": gender,
    "voted": voted
})

# Normalize for nflows (mean 0, std 1)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Convert to numpy for nflows input
data_for_nflows = df_scaled.to_numpy().astype(np.float32)

# Ready to use in PyTorch / nflows
print(data_for_nflows.shape)  # should be (1000, 5)
plt.scatter(voted, education)
plt.show(block = False)

df = np.array(df)

num_layers = 10
base_dist = StandardNormal(shape=[2])

transforms = []
for _ in range(num_layers):
    transforms.append(ReversePermutation(features=2))
    transforms.append(MaskedAffineAutoregressiveTransform(features=2, 
                                                          hidden_features=4))
transform = CompositeTransform(transforms)

flow = Flow(transform, base_dist)
optimizer = optim.Adam(flow.parameters())

num_iter = 10000
for i in range(num_iter):
    idx = np.random.choice(df.shape[0], 1000, replace=True)
    x = df[idx, :][:, [1, 3]]
    # x = df[:, [1, 3]]
    x = torch.tensor(x, dtype=torch.float32)
    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=x).mean()
    loss.backward()
    optimizer.step()
    
    # if (i + 1) % 500 == 0:
        xline = torch.linspace(22, 200, 100)
        yline = torch.linspace(-0.2, 1.2, 100)
        xgrid, ygrid = torch.meshgrid(xline, yline)
        xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

        with torch.no_grad():
            zgrid = flow.log_prob(xyinput).exp().reshape(100, 100)

        plt.contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy())
        # plt.title('iteration {}'.format(i + 1))
        plt.show(block=False)

plt.scatter(df[:, 1], df[:, 3], alpha=0.2)
plt.show(block=False)
