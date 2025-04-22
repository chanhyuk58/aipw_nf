import pandas as pd
import numpy as np
import normflows as nf
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from sklearn.preprocessing import StandardScaler
# import random

# Get device to be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define target
target = nf.distributions.target.ConditionalDiagGaussian()
context_size = 4

# Plot target
grid_size = 100
xx, yy = torch.meshgrid(torch.linspace(-2, 2, grid_size), torch.linspace(-2, 2, grid_size), indexing='ij')
zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
zz = zz.to(device)
context_plot = torch.cat([torch.tensor([0.3, 0.9]).to(device) + torch.zeros_like(zz), 
                          0.6 * torch.ones_like(zz)], dim=-1)
logp = target.log_prob(zz, context_plot)
p_target = torch.exp(logp).view(*xx.shape).cpu().data.numpy()

plt.figure(figsize=(10, 10))
plt.pcolormesh(xx, yy, p_target, shading='auto')
plt.gca().set_aspect('equal', 'box')
plt.show(block=False)

# Define flows
K = 4

latent_size = 2
hidden_units = 128
num_blocks = 2

flows = []
for i in range(K):
    flows += [nf.flows.MaskedAffineAutoregressive(latent_size, hidden_units, 
                                                  context_features=context_size, 
                                                  num_blocks=num_blocks)]
    flows += [nf.flows.LULinearPermute(latent_size)]

# Set base distribution
q0 = nf.distributions.DiagGaussian(2, trainable=False)
    
# Construct flow model
model = nf.ConditionalNormalizingFlow(q0, flows, target)

# Move model on GPU if available
model = model.to(device)

# Plot initial flow distribution, target as contours
model.eval()
log_prob = model.log_prob(zz, context_plot).to('cpu').view(*xx.shape)
model.train()
prob = torch.exp(log_prob)
prob[torch.isnan(prob)] = 0

plt.figure(figsize=(10, 10))
plt.pcolormesh(xx, yy, prob.data.numpy(), shading='auto')
plt.contour(xx, yy, p_target, cmap=plt.get_cmap('cool'), linewidths=2)
plt.gca().set_aspect('equal', 'box')
plt.show(block=False)

# Train model
max_iter = 5000
batch_size= 128

loss_hist = np.array([])

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)


for it in tqdm(range(max_iter)):
    optimizer.zero_grad()
    
    # Get training samples
    context = torch.cat([torch.randn((batch_size, 2), device=device), 
                         0.5 + 0.5 * torch.rand((batch_size, 2), device=device)], 
                        dim=-1)
    x = target.sample(batch_size, context)
    
    # Compute loss
    loss = model.forward_kld(x, context)
    
    # Do backprop and optimizer step
    if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        optimizer.step()
    
    # Log loss
    loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())

# Plot loss
plt.figure(figsize=(10, 10))
plt.plot(loss_hist, label='loss')
plt.legend()
plt.show(block=False)


# Plot trained flow distribution, target as contours
model.eval()
log_prob = model.log_prob(zz, context_plot).to('cpu').view(*xx.shape)
model.train()
prob = torch.exp(log_prob)
prob[torch.isnan(prob)] = 0

plt.figure(figsize=(10, 10))
plt.pcolormesh(xx, yy, prob.data.numpy(), shading='auto')
plt.contour(xx, yy, p_target, cmap=plt.get_cmap('cool'), linewidths=2)
plt.gca().set_aspect('equal', 'box')
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
