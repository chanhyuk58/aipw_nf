# Import packages
import numpy as np
import torch
import normflows as nf
import larsflow as lf
from matplotlib import pyplot as plt
from tqdm import tqdm

# Get device
enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')

# Function for model creation {{{
def create_model(base='gauss'):
    # Set up model

    # Define flows
    K = 8 # number of layers
    torch.manual_seed(10)

    latent_size = 2
    hidden_units = 128
    num_blocks = 2
    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
    flows = []
    for i in range(K):
        flows += [nf.flows.MaskedAffineAutoregressive(latent_size, hidden_units, 
                                                  context_features=context_size, 
                                                  num_blocks=num_blocks)]
        flows += [nf.flows.LULinearPermute(latent_size)]
        # param_map = nf.nets.MLP([latent_size // 2, 32, 32, latent_size], init_zeros=True)
        # flows += [nf.flows.AffineCouplingBlock(param_map)]
        # flows += [nf.flows.Permute(latent_size, mode='swap')]
        # flows += [nf.flows.ActNorm(latent_size)]

    # Set prior and q0
    if base == 'resampled':
        a = nf.nets.MLP([latent_size, 256, 256, 256, 1], output_fn="sigmoid")
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
    # model = lf.NormalizingFlow(q0=q0, flows=flows, p=p)
    # Construct flow model
    model = nf.ConditionalNormalizingFlow(q0, flows)

    # Move model on GPU if available
    return model.to(device)
# }}}

# Function to train model {{{
def train(model, max_iter=2000, num_samples=1000, lr=1e-3, weight_decay=1e-3, 
          q0_weight_decay=1e-4):
    # Do mixed precision training
    optimizer = torch.optim.Adam(model.parameters(),  lr=lr, weight_decay=weight_decay)
    model.train()

    for it in tqdm(range(max_iter)):
        
        # Get training samples
        context = torch.cat([torch.randn((num_samples, 1), device=device), 
                         0.5 + 0.5 * torch.rand((num_samples, 1), device=device)], 
                        dim=-1)
        # x = model.p.sample(num_samples, context) # Sample from target distribution p
        # x = nf.distributions.TwoMoons().sample(num_samples) + context # Sample from target distribution p
        # x = nf.distributions.CircularGaussianMixture().sample(num_samples)+ context # Sample from target distribution p
        x = nf.distributions.RingMixture().sample(num_samples)+ context # Sample from target distribution p

        loss = model.forward_kld(x, context) # loss function is defined with KL divergence
        # loss = model.reverse_kld(x, context) # loss function is defined with KL divergence

        loss.backward()
        optimizer.step()

        # Clear gradients
        nf.utils.clear_grad(model)
# }}}

# Plot function {{{
def plot_results(model, target=True, a=False, base=False, save=False, prefix=''):
    # Prepare z grid for evaluation
    grid_size = 300
    xx, yy = torch.meshgrid(torch.linspace(-3, 3, grid_size), torch.linspace(-3, 3, grid_size))
    zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
    zz = zz.to(device)
    # context_plot = torch.cat([torch.tensor([0.3, 0.9]).to(device) + torch.zeros_like(zz), 
    #                       0.6 * torch.ones_like(zz)], dim=-1)
    context_plot = torch.cat([torch.tensor([0.3, 0.9]).to(device) + torch.zeros_like(zz)], dim=-1)
    # context_plot = torch.cat([torch.randn((zz.size()[0], 1), device=device), 
    #                      0.5 + 0.5 * torch.rand((zz.size()[0], 1), device=device)], 
    #                     dim=-1)
    
    # log_prob = model.p.log_prob(zz, context_plot).to('cpu').view(*xx.shape)
    # prob = torch.exp(log_prob)
    # prob[torch.isnan(prob)] = 0
    # prob_target = prob.data.numpy()
    
    if target:
        plt.figure(figsize=(15, 15))
        plt.pcolormesh(xx, yy, prob_target)
        #cs = plt.contour(xx, yy, prob_target, [.025, .15, .7], colors='w', linewidths=3)#, linestyles='dashed')
        #for c in cs.collections:
        #    c.set_dashes([(0, (10.0, 10.0))])
        plt.gca().set_aspect('equal', 'box')
        plt.axis('off')
        if save:
            plt.savefig(prefix + 'target.png', dpi=300)
        plt.show(block=False)

    model.eval()
    log_prob = model.log_prob(zz, context_plot).to('cpu').view(*xx.shape)

    prob = torch.exp(log_prob)
    prob[torch.isnan(prob)] = 0
    prob_model = prob.data.numpy()

    plt.figure(figsize=(15, 15))
    plt.pcolormesh(xx, yy, prob_model)#, cmap=plt.get_cmap('coolwarm'))
    #cs = plt.contour(xx, yy, prob_model, [.025, .2, .35], colors='w', linewidths=3)#, linestyles='dashed')
    #for c in cs.collections:
    #    c.set_dashes([(0, (10.0, 10.0))])
    plt.gca().set_aspect('equal')
    plt.axis('off')
    if save:
        plt.savefig(prefix + 'model.png', dpi=300)
    plt.show(block=False)

    if base:
        log_prob = model.q0.log_prob(zz, context_plot).to('cpu').view(*xx.shape)
        prob = torch.exp(log_prob)
        prob[torch.isnan(prob)] = 0
        prob_base = prob.data.numpy()

        plt.figure(figsize=(15, 15))
        plt.pcolormesh(xx, yy, prob_base)
        #cs = plt.contour(xx, yy, prob_base, [.025, .075, .135], colors='w', linewidths=3)#, linestyles='dashed')
        #for c in cs.collections:
        #    c.set_dashes([(0, (10.0, 10.0))])
        plt.gca().set_aspect('equal')
        plt.axis('off')
        if save:
            plt.savefig(prefix + 'base.png', dpi=300)
        plt.show(block=False)
    
    if a:
        prob = model.q0.a(zz).to('cpu').view(*xx.shape)
        prob[torch.isnan(prob)] = 0

        plt.figure(figsize=(15, 15))
        plt.pcolormesh(xx, yy, prob.data.numpy())
        plt.gca().set_aspect('equal')
        plt.axis('off')
        if save:
            plt.savefig(prefix + 'a.png', dpi=300)
        plt.show(block=False)
    
    # # Compute KLD
    # eps = 1e-10
    # kld = np.sum(prob_target * np.log((prob_target + eps) / (prob_model + eps)) * 6 ** 2 / grid_size ** 2)
    # print('KL divergence: %f' % kld)
# }}}

# Train models
p = [nf.distributions.TwoMoons(), nf.distributions.CircularGaussianMixture(), nf.distributions.RingMixture(), nf.distributions.target.ConditionalDiagGaussian()]
name = ['moons', 'circle', 'rings', 'conditional']

context_size = 2

nf.distributions.target.ConditionalDiagGaussian().sample(num_samples=1000, context=context)

for i in range(len(p)):
    i = 3
    # Train model with Gaussain base distribution
    model = create_model(p[i], 'gauss')
    train(model, max_iter=2000)
    # Plot and save results
    plot_results(model, save=False,
                 prefix='results/2d_distributions/fkld/rnvp/' 
                 + name[i] + '_gauss_')
    
    # Train model with mixture of Gaussians base distribution
    model = create_model(p[i], 'gaussian_mixture')
    train(model)
    # Plot and save results
    plot_results(model, save=False,
                 prefix='results/2d_distributions/fkld/rnvp/' 
                 + name[i] + '_gaussian_mixture_')
    
    # Train model with resampled base distribution
    # model = create_model(p[i], 'resampled')
    model = create_model('resampled')
    train(model, max_iter=500)
    # Plot and save results
    plot_results(model, save=True, a=False, target=False, base=True,
                 prefix='../figures/cond_ring_resampled_')
