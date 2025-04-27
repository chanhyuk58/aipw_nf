# Import packages
import numpy as np
import torch
import normflows as nf
import larsflow as lf
from matplotlib import pyplot as plt
from tqdm import tqdm

# Get device
enable_cuda = True
enable_mps = False # Using GPU in Apple Silicon Mac. Slower than CPU?
device = torch.device(
    'cuda' if torch.cuda.is_available() and enable_cuda else
    'mps' if torch.backends.mps.is_available() and enable_mps else
    'cpu'
)
print(torch.__config__.parallel_info())

# Generate Target Population {{{
N = 10**5
np.random.seed(63130)

# Generate Covariates
x = torch.cat([
    torch.randn((N, 1), device=device), 
    0.5 + 0.5 * torch.rand((N, 1), device=device),
    0.5 * torch.rand((N, 1), device=device),
    0.2 * torch.rand((N, 1), device=device),
    0.1 * torch.rand((N, 1), device=device),
], 
              dim=-1)

# Generate the distribution for the error term = y | x
# n_modes=3
# dim=1
# d = nf.distributions.GaussianMixture(n_modes=3, dim=1, trainable=True,
#                                       loc=np.array([[-4.0], [1.0], [6.0]]),
#                                       scale=np.array([[2.0], [1.2], [3.0]])).sample(N).float()

## 2D
# target = nf.distributions.TwoMoons()
target = nf.distributions.RingMixture()
# target = nf.distributions.CircularGaussianMixture()
d = target.sample(N)

# y = d + x[:,0].view(N, 1) + x[:,1].view(N, 1)

# Generate Target Sample
sample_size = 1000
sample_idx = np.random.choice(range(N), sample_size)

xn = x[sample_idx].to(device)
dn = d[sample_idx].to(device)

yn = torch.cat([
    dn[:,0].view(sample_size, 1) + xn.sum(dim=1).view(sample_size, 1), 
    dn[:,1].view(sample_size, 1) + xn.sum(dim=1).view(sample_size, 1), 
], dim=-1)
# }}}

# Function for model creation {{{
def create_model(base='resampled', outcome=None, covariates=None):
    # Set up model

    # Define flows
    K = 4 # number of layers
    torch.manual_seed(10)

    latent_size = outcome.size()[1] # dimension of the latent space
    context_size = covariates.size()[1] # dimension of the context / covariates space
    hidden_units = 128 # hidden_units for MaskedAffineAutoregressive
    num_blocks = 2 # block for MaskedAffineAutoregressive
    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
    flows = []
    for i in range(K):
        flows += [nf.flows.MaskedAffineAutoregressive(latent_size, hidden_units, 
                                                  context_features=context_size, 
                                                  num_blocks=num_blocks)]
        flows += [nf.flows.LULinearPermute(latent_size)]

    # Set prior and q0
    if base == 'resampled':
        # a = nf.nets.MLP([latent_size, 128, 256, 512, 256, 128, 1], output_fn="sigmoid") # 5 layers
        a = nf.nets.MLP([latent_size, 256, 256, 1], output_fn="sigmoid") # layers
        q0 = lf.distributions.ResampledGaussian(latent_size, a, 100, 0.1, trainable=False)
    elif base == 'gaussian_mixture':
        n_modes = 10
        q0 = nf.distributions.GaussianMixture(n_modes, latent_size, trainable=True,
                                              loc=(np.random.rand(n_modes, latent_size) - 0.5) * 5,
                                              scale=0.5 * np.ones((n_modes, latent_size)))
    elif base == 'gauss':
        q0 = nf.distributions.DiagGaussian(latent_size, trainable=False)
    else:
        raise NotImplementedError('This base distribution is not implemented.')

    # Construct flow model
    model = nf.ConditionalNormalizingFlow(q0, flows)

    # Move model on GPU if available
    return model.to(device)
# }}}

# Function to train model {{{
def train(model, covariates=None, outcome=None, max_iter=2000, sample_size=1000, lr=1e-3, weight_decay=1e-3, 
          q0_weight_decay=1e-4):
    # Do mixed precision training
    optimizer = torch.optim.Adam(model.parameters(),  lr=lr, weight_decay=weight_decay)
    model.train()

    global loss_hist

    for it in tqdm(range(max_iter)):
        # Clear gradients
        nf.utils.clear_grad(model)
        optimizer.zero_grad()

        # Bootstrap training sample
        boot_idx = np.random.choice(range(sample_size), sample_size, replace=True)
        b_covariates = covariates[boot_idx]
        b_outcome = outcome[boot_idx]

        # b_covariates = xn[boot_idx]
        # b_outcome = yn[boot_idx]

        loss = model.forward_kld(x=b_outcome, context=b_covariates) # loss function is defined with forward KL divergence
        # loss = model.reverse_kld(b_outcome, b_covariates) # loss function is defined with reverse KL divergence

        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward(retain_graph=True)
            optimizer.step()
            # Log loss

        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
# }}}

# Plot function {{{
def plot_results(model, dim=1, a=False, base=False, save=False, prefix=''):
    # Prepare z grid for evaluation
    grid_size = 200
    if dim == 1:
        zz = torch.linspace(d.min(), d.max(), grid_size) # for 1d 
        zz = zz.to(device)
        # zz = zz.view(zz.size()[0], 1)
        context_plot = torch.tensor([0.0, 0.5, 0.0, 0.0, 0.0]).to(device).repeat(zz.size()[0], 1)
        # context_plot = torch.tensor([0.1, 0.1]).to(device).repeat(zz.size()[0], 1)
        model.eval()
        log_prob = model.log_prob(zz, context_plot).to('cpu')
        # log_prob = model_resampled.log_prob(zz, context_plot).to('cpu')
        prob = torch.exp(log_prob).view(*zz.shape)
        plt.plot(zz, prob.detach().numpy())
    elif dim==2:
        xx, yy = torch.meshgrid(torch.linspace(-5, 5, grid_size), torch.linspace(-5, 5, grid_size))
        zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2) # for 2d
        zz = zz.to(device)
        # zz = zz.view(zz.size()[0], 1)
        # context_plot = torch.tensor([-0.05, 0.05, 0.15, -0.05, 0.05]).to(device).repeat(zz.size()[0], 1)
        context_plot = xn.median(dim=0).values.to(device).repeat(zz.size()[0], 1)
        
        # context_plot = torch.tensor([0.1, 0.1]).to(device).repeat(zz.size()[0], 1)
        model.eval()
        log_prob = model.log_prob(zz, context_plot).to('cpu')
        # log_prob = model_resampled.log_prob(zz, context_plot).to('cpu')
        prob = torch.exp(log_prob).view(*xx.shape)
        prob[torch.isnan(prob)] = 0
        prob_model = prob.data.numpy()

        # plt.figure(figsize=(15, 15))
        plt.pcolormesh(xx, yy, prob_model)
    else:
        print('error: check the input dimension')

    if save:
        plt.savefig(prefix + 'model.pdf', dpi=300)
    plt.show(block=False)

    if base:
        log_prob = model.q0.log_prob(zz, context_plot).to('cpu').view(*zz.shape)
        prob = torch.exp(log_prob)
        prob[torch.isnan(prob)] = 0
        prob_base = prob.data.numpy()

        # plt.figure(figsize=(15, 15))
        plt.plot(zz, prob_base)
        # plt.pcolormesh(xx, yy, prob_base)
        if save:
            plt.savefig(prefix + 'base.pdf', dpi=300)
        plt.show(block=False)
    
    if a:
        prob = model.q0.a(zz).to('cpu').view(*xx.shape)
        prob[torch.isnan(prob)] = 0
        prob_a = prob.data.numpy()
        # plt.figure(figsize=(15, 15))
        plt.plot(zz, prob_a)
        # plt.pcolormesh(xx, yy, prob_a)
        if save:
            plt.savefig(prefix + 'a.pdf', dpi=300)
        plt.show(block=False)
    
    # # Compute KLD
    # eps = 1e-10
    # kld = np.sum(prob_target * np.log((prob_target + eps) / (prob_model + eps)) * 6 ** 2 / grid_size ** 2)
    # print('KL divergence: %f' % kld)
# }}}

# Train model with resampled base distribution
model_resampled = create_model(base='resampled', outcome=yn, covariates=xn)
loss_hist = np.array([])
train(model_resampled, outcome=yn, covariates=xn, max_iter=10000)


# Plot loss
plt.plot(loss_hist, label='loss')
plt.legend()
plt.show()

# Plot and save results
plot_results(model_resampled, dim=2, save=True, a=False, base=False,
             prefix='../figures/10000_cond_rings_resampled_')

# Plot 1D target
# plt.hist(d.detach().numpy(), density=True, bins=500)

# Plot 2D target
grid_size = 200
xx, yy = torch.meshgrid(torch.linspace(-5, 5, grid_size), torch.linspace(-5, 5, grid_size), indexing='ij')
zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
zz = zz.to(device)
logp = target.log_prob(zz)
p_target = torch.exp(logp).view(*xx.shape).cpu().data.numpy()

plt.pcolormesh(xx, yy, p_target)

# Save and show
plt.savefig('../figures/10000_cond_rings_truth.pdf')
plt.show(block=False)
