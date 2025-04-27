import pandas as pd
import numpy as np
import torch
import normflows as nf
import larsflow as lf
from matplotlib import pyplot as plt
from tqdm import tqdm
import scipy.stats
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import gc

from npDoseResponse import DRCurve
from npDoseResponse import DRDerivCurve, NeurNet
import sys
import pickle

import Supplement

job_id = int(63130)
print(job_id)

# interactive plotting
plt.ioff()
# plt.ion()

#=======================================================================================#

# Compute the true conditional densities at the sample points
def CondDenTrue(Y, X):
    res = np.exp(-(Y - scipy.stats.norm.cdf(3*np.dot(X, theta)) + 1/2)**2/(2*0.75**2))/(0.75*np.sqrt(2*np.pi))
    return res

# Set up for NF
# Get device
enable_cuda = True
enable_mps = False # Using GPU in Apple Silicon Mac. Slower than CPU?
device = torch.device(
    'cuda' if torch.cuda.is_available() and enable_cuda else
    'mps' if torch.backends.mps.is_available() and enable_mps else
    'cpu'
)

# Function for flow model creation {{{
def flow_model(base='resampled', outcome=None, contexts=None):
    # Set up model
    '''
    This is a function to create a flow model to estimate P(Y | X)
    base: the base distribution
            - 'resampled': resampled Gaussian
            - 'gauss': Gaussian
            - 'gausssian_mixture': Gaussian mixture
    outcome: Y
    contexts: X
    '''

    # Define flows
    K = 32 # number of layers for the flow
    torch.manual_seed(10)

    latent_size = outcome.size()[1] # dimension of the latent space
    context_size = contexts.size()[1] # dimension of the context / contexts space
    hidden_units = 64 # hidden_units for MaskedAffineAutoregressive
    num_blocks = 2 # block for MaskedAffineAutoregressive

    flows = []
    for i in range(K):
        flows += [nf.flows.MaskedAffineAutoregressive(latent_size, hidden_units, 
                                                  context_features=context_size, 
                                                  num_blocks=num_blocks)]
        flows += [nf.flows.LULinearPermute(latent_size)]

    # Set prior and q0
    if base == 'resampled':
        a = nf.nets.MLP([latent_size, 256, 256, 1], output_fn='sigmoid') # 2 layers
        # a = nf.nets.MLP([latent_size, 256, 256, 1], output_fn='sigmoid') # 2 layers
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

# Function to train a flow model {{{
def train(model, outcome=None, contexts=None, boot_iter=100000, sample_size=1000, 
          lr=1e-3, weight_decay=1e-3, q0_weight_decay=1e-4, tol=0.03):
    # Do mixed precision training
    optimizer = torch.optim.Adam(model.parameters(),  lr=lr, weight_decay=weight_decay)
    model.train()

    global loss_hist
    iters = tqdm(range(boot_iter))
    for it in iters:
        # Clear gradients
        nf.utils.clear_grad(model)
        optimizer.zero_grad()

        # Bootstrap training sample
        boot_idx = np.random.choice(range(sample_size), sample_size, replace=True)
        b_contexts = contexts[boot_idx]
        b_outcome = outcome[boot_idx]

        # b_contexts = xn[boot_idx]
        # b_outcome = yn[boot_idx]

        loss = model.forward_kld(x=b_outcome, context=b_contexts) # loss function is defined with forward KL divergence
        # loss = model.reverse_kld(b_outcome, b_contexts) # loss function is defined with reverse KL divergence

        if ~(torch.isnan(loss) | torch.isinf(loss)):
            if it > 0:
                if ((abs(loss_hist.min() - loss.detach().numpy()) <= tol) | (loss.detach().numpy() < 1)):
                    print('The loss has reached the good enough level. \nCurrent Loss: ' + str(loss) + '\nCurrent iteration: ' + str(it))
                    break
            loss.backward(retain_graph=True)
            optimizer.step()
            iters.set_postfix({'loss': str(loss.detach().numpy())})

        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
# }}}

#=======================================================================================#

# reg_mod_lst = ['LR', 'RF', 'NN']
reg_mod_lst = ['NN']
cond_type = ['true', 'kde', 'nf']
fac_lst = [0.75, 1, 1.25, 1.5, 2]
n_lst = [500, 1000, 2000]

# Start the loop {{{
regiter = tqdm(reg_mod_lst)
for reg in regiter:
    # reg = 'NN'
    faciter = tqdm(fac_lst)
    for fac in faciter:
        conditer = tqdm(cond_type)
        for cond in conditer:
        # cond = 'true'
            niter = tqdm(n_lst)
            for n in niter:
            # n = 1000
                gc.collect()
                rho = 0.5  # correlation between adjacent Xs
                if n == 10000:
                    d = 100   # Dimension of the confounding variables
                    model_nn2_n10000 = Supplement.NeuralNet2_n10000(k=d, 
                                                          lr=0.4,
                                                          momentum = 0.0, 
                                                          epochs=100,
                                                          weight_decay=0.075)
                    cond_reg_mod = model_nn2_n10000
                else:
                    d = 20   # Dimension of the confounding variables
                    model_nn2_n1000 = Supplement.NeuralNet2_n1000(k=d, 
                                                  lr=0.01,
                                                  momentum = 0.9, 
                                                  epochs=100,
                                                  weight_decay=0.3)
                    cond_reg_mod = model_nn2_n1000
                

                Sigma = np.zeros((d,d)) + np.eye(d)
                for i in range(d):
                    for j in range(i+1, d):
                        if (j < i+2) or (j > i+d-2):
                            Sigma[i,j] = rho
                            Sigma[j,i] = rho
                sig = 1

                np.random.seed(job_id)
                # Data generating process
                X_sim = np.random.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=n)
                nu = np.random.randn(n)
                eps = np.random.randn(n)

                theta = 1/(np.linspace(1, d, d)**2)
                # theta = np.linspace(1, 1, d)

                T_sim = scipy.stats.norm.cdf(3*np.dot(X_sim, theta)) + 3*nu/4 - 1/2
                Y_sim = 1.2*T_sim + T_sim**2 + T_sim*X_sim[:,0] + 1.2*np.dot(X_sim, theta) + eps*np.sqrt(0.5+ scipy.stats.norm.cdf(X_sim[:,0]))
                X_dat = np.column_stack([T_sim, X_sim])
            
                
                
                if reg == 'LR':
                    reg_mod = LinearRegression()
                elif reg == 'RF':
                    reg_mod = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=1)
                elif reg == 'NN':
                    reg_mod = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', 
                                           learning_rate='adaptive', learning_rate_init=0.1, 
                                           random_state=1, max_iter=200)
                # Bandwidth choice
                h = fac*np.std(T_sim)*n**(-1/5)
                
                t_qry = np.linspace(-2, 2, 81)

                if cond == 'true':
                    # True conditional density values
                    true_cond = CondDenTrue(T_sim, X_sim)
                    cond_mod = true_cond
                elif cond == 'nf':
                    # convert to tensor
                    T_sim_tensor = torch.from_numpy(T_sim).view(n, 1).to(device)
                    X_sim_tensor = torch.from_numpy(X_sim).to(device)

                    # set up flow model and train
                    flowTS = flow_model(base='resampled', outcome=T_sim_tensor, contexts=X_sim_tensor)
                    loss_hist = np.array([])
                    train(model=flowTS, outcome=T_sim_tensor, contexts=X_sim_tensor, sample_size=n, tol=1e-4)

                    # check and plot the loss
                    loss_hist.min()
                    plt.clf()
                    plt.plot(loss_hist, label='loss')
                    plt.savefig('../figures/loss.pdf')
                    plt.show(block=False)

                    # calculate the conditional probability for the sample
                    log_prob = flowTS.log_prob(x=T_sim_tensor, context=X_sim_tensor).to('cpu')
                    prob = torch.exp(log_prob)
                    # prob = torch.exp(log_prob).view(*T_sim_tensor.shape)
                    prob[torch.isnan(prob)] = 0
                    nf_cond = prob.detach().numpy()
                    cond_mod = nf_cond

                    # plot the conditional prob for the sample
                    plt.clf()
                    plt.bar(T_sim, nf_cond)
                    plt.savefig('../figures/nf.pdf', dpi=300)
                    plt.show(block=False)

                elif cond == 'kde':
                    # cond = 'kde'
                    # Conditional density model
                    regr_nn2 = MLPRegressor(hidden_layer_sizes=(20,), activation='relu', learning_rate='adaptive', 
                                    learning_rate_init=0.1, random_state=1, max_iter=200)
                    cond_mod = regr_nn2
                else:
                    cond_mod = cond_reg_mod
                    
                # if cond == 'nf':
                #     m_est_dr1, sd_est_dr1 = DRCurve(Y=Y_sim, X=X_dat, t_eval=t_qry, est='DR', 
                #                                     mu=reg_mod, condTS_type='true', condTS_mod=cond_mod, 
                #                                     tau=0.001, L=1, h=h, kern='epanechnikov', h_cond=None, 
                #                                     print_bw=False, self_norm=True)
                # else:
                #     m_est_dr1, sd_est_dr1 = DRCurve(Y=Y_sim, X=X_dat, t_eval=t_qry, est='DR', 
                #                                     mu=reg_mod, condTS_type=cond, condTS_mod=cond_mod, 
                #                                     tau=0.001, L=1, h=h, kern='epanechnikov', h_cond=None, 
                #                                     print_bw=False, self_norm=True)
                #
                # with open('./Results/Simulation2_m_est'+str(job_id)+'_'+str(reg)+'_h'+str(fac)+'_condmod_'+str(cond)+'_n_'+str(n)+'_selfnorm.dat', 'wb+') as file:
                #     pickle.dump([m_est_dr1, sd_est_dr1], file)
                
                if reg == 'LR':
                    reg_mod = LinearRegression()
                elif reg == 'RF':
                    reg_mod = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=1)
                elif reg == 'NN':
                    reg_mod = NeurNet

                if cond == 'nf':
                    theta_dr1, theta_sd1 = DRDerivCurve(Y=Y_sim, X=X_dat, t_eval=t_qry, est='DR', 
                                                                beta_mod=reg_mod, n_iter=1000, 
                                                      lr=0.01, condTS_type='true', condTS_mod=cond_mod, 
                                                      tau=0.001, L=1, h=h, kern='epanechnikov', h_cond=None, 
                                                      print_bw=True, delta=0.01, self_norm=True)
                else:
                    theta_dr1, theta_sd1 = DRDerivCurve(Y=Y_sim, X=X_dat, t_eval=t_qry, est='DR', 
                                                                beta_mod=reg_mod, n_iter=1000, 
                                                      lr=0.01, condTS_type=cond, condTS_mod=cond_mod, 
                                                      tau=0.001, L=1, h=h, kern='epanechnikov', h_cond=None, 
                                                      print_bw=True, delta=0.01, self_norm=True)

                with open('./Results/Simulation2_theta_est'+str(job_id)+'_'+str(reg)+'_h'+str(fac)+'_condmod_'+str(cond)+'_n_'+str(n)+'_selfnorm.dat', 'wb+') as file:
                    # pickle.dump([theta_dr5, theta_sd5, theta_dr1, theta_sd1], file)
                    pickle.dump([theta_dr1, theta_sd1], file)
# }}}        

#=======================================================================================#

# Define 
def Bias(est_mat, true_val):
    a = est_mat - true_val
    return np.mean(a, axis=0, where=~(np.isinf(a) | np.isnan(a)))

def RMSE(est_mat, true_val):
    a = (est_mat - true_val)**2
    return np.sqrt(np.mean(a, axis=0, where=~(np.isinf(a) | np.isnan(a))))

def CovProb(est_mat, sd_mat, true_val, alpha=0.95):
    low_ci = est_mat + sd_mat*scipy.stats.norm.ppf((1 - alpha)/2)
    up_ci = est_mat - sd_mat*scipy.stats.norm.ppf((1 - alpha)/2)
    return np.mean((low_ci <= true_val) & (up_ci >= true_val), axis=0)

# Create the sample
rho = 0.5  # correlation between adjacent Xs
d = 20   # Dimension of the confounding variables
n = 2000

Sigma = np.zeros((d,d)) + np.eye(d)
for i in range(d):
    for j in range(i+1, d):
        if (j < i+2) or (j > i+d-2):
            Sigma[i,j] = rho
            Sigma[j,i] = rho
sig = 1

job_id = 63130
np.random.seed(job_id)
# Data generating process
X_sim = np.random.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=n)
nu = np.random.randn(n)
eps = np.random.randn(n)

theta = 1/(np.linspace(1, d, d)**2)

T_sim = scipy.stats.norm.cdf(3*np.dot(X_sim, theta)) + 3*nu/4 - 1/2
Y_sim = 1.2*T_sim + T_sim**2 + T_sim*X_sim[:,0] + 1.2*np.dot(X_sim, theta) + eps*np.sqrt(0.5+ scipy.stats.norm.cdf(X_sim[:,0]))
X_dat = np.column_stack([T_sim, X_sim])

t_qry = np.linspace(-2, 2, 81)

# True values of m(t) and theta(t)
m_true = 1.2*t_qry + t_qry**2
theta_true = 1.2 + 2*t_qry

# Produce figures
fac_lst = [0.75, 1, 1.25, 1.5, 2]
cond_type = ['true', 'kde', 'nf']
n_lst = [500, 1000, 2000]
model_lst = ['nn']

# Plot {{{
j = 0
plt.rcParams.update({'font.size': 23})
fig, ax = plt.subplots(2, 3, figsize=(26, 16))
niter = tqdm(n_lst)
for n in niter:
    conditer = tqdm(cond_type)
    for cond in conditer:
        print(cond)
        with open('./Results/Simulation2_theta_est'+str(job_id)+'_'+str(reg)+'_h'+str(fac)+'_condmod_'+str(cond)+'_n_'+str(n)+'_selfnorm.dat', "rb") as file:
            theta_dr1_lst, sd_dr1_lst = pickle.load(file)
        
        # Bias
        theta_dr1_bias = Bias([np.array(theta_dr1_lst)], theta_true)
        
        # RMSE
        theta_dr1_rmse = RMSE([np.array(theta_dr1_lst)], theta_true)
        
        # Coverage
        theta_dr1_cov = CovProb([np.array(theta_dr1_lst)], np.array(sd_dr1_lst), theta_true, alpha=0.95)
        
        if cond == 'true':
            res_bias = np.column_stack([t_qry, theta_dr1_bias])
            res_rmse = np.column_stack([t_qry, theta_dr1_rmse])
            res_cov = np.column_stack([t_qry, theta_dr1_cov])
        else:
            res_bias = np.column_stack([res_bias, theta_dr1_bias])
            res_rmse = np.column_stack([res_rmse, theta_dr1_rmse])
            res_cov = np.column_stack([res_cov, theta_dr1_cov])
                                           
    
    res_bias2 = pd.DataFrame(res_bias)
    res_bias2.columns = ['Query point', 'DR (true, L=1)', 'DR (Residual KDE, L=1)', 'DR (nf, L=1)']
    
    res_rmse2 = pd.DataFrame(res_rmse)
    res_rmse2.columns = ['Query point', 'DR (true, L=1)', 'DR (Residual KDE, L=1)', 'DR (nf, L=1)']
    
    res_cov2 = pd.DataFrame(res_cov)
    res_cov2.columns = ['Query point', 'DR (true, L=1)', 'DR (Residual KDE, L=1)', 'DR (nf, L=1)']
    
    # Plotting bias
    ax[0][j].axhline(y=0, color='black', linestyle='dashed', linewidth=5, alpha=0.5)
    col_lst = ['tab:cyan', 'red', 'green', 'blue', 'grey']
    col_lab = [r'$\widehat{\theta}_{\mathrm{DR}}(t)$ (True)', r'$\widehat{\theta}_{\mathrm{DR}}(t)$ (Residual KDE)', r'$\widehat{\theta}_{\mathrm{DR}}(t)$ (NF)']
    mark_lst = ["o", "v", "^", "<", "P", "X", ">"]
    # col_lst2 = ['tab:brown', 'darkorange', 'darkblue']
    # for i in range(res_bias2.shape[1] - 1):
    #     ax[0][j].plot(t_qry, res_bias2.iloc[:,i+1], markersize=7, linewidth=3, marker=mark_lst[i+4], 
    #                   label=res_bias2.columns[i+1], color=col_lst2[i], alpha=0.8)
    for i in range(res_bias2.shape[1] - 1):
        ax[0][j].plot(t_qry, res_bias2.iloc[:,i+1], markersize=7, linewidth=3, marker=mark_lst[i], 
                      label=col_lab[i], color=col_lst[i], alpha=0.8)
    # ax[0][j].set_xlabel(r'Query point $T=t$')
    ax[0][0].set_ylabel(r'Bias for $\theta(t)$')
    ax[0][j].set_title(r'$n=$'+str(n))
    # ax[0][3].legend(bbox_to_anchor=(1, 0.9))
    
    # Plotting RMSE
    # for i in range(res_rmse2.shape[1] - 1):
    #     ax[1][j].plot(t_qry, res_rmse2.iloc[:,i+1], linewidth=3, markersize=8, marker=mark_lst[i+4], 
    #                   label=res_rmse2.columns[i+1], color=col_lst2[i], alpha=0.8)
    for i in range(res_rmse2.shape[1] - 1):
        ax[1][j].plot(t_qry, res_rmse2.iloc[:,i+1], linewidth=3, markersize=7, marker=mark_lst[i], 
                      label=col_lab[i], color=col_lst[i], alpha=0.8)
    # ax[1][j].set_xlabel(r'Query point $T=t$')
    ax[1][j].set_ylim([-0.5, 7])
    ax[1][0].set_ylabel(r'RMSE for $\theta(t)$')
    ax[1][1].legend(bbox_to_anchor=(1.2, -0.1), fontsize=20, ncol=len(cond_type))
    
    # Plotting Coverage
    # ax[2][j].axhline(y=0.95, color='black', linestyle='dashed', linewidth=5, alpha=0.5, label='Nominal level')
    # # for i in range(res_cov2.shape[1] - 1):
    # #     ax[2][j].plot(t_qry, res_cov2.iloc[:,i+1], linewidth=3, markersize=7, marker=mark_lst[i+4], 
    # #                   label=res_cov2.columns[i+1], color=col_lst2[i])
    # for i in range(res_cov2.shape[1] - 1):
    #     ax[2][j].plot(t_qry, res_cov2.iloc[:,i+1], linewidth=3, markersize=7, marker=mark_lst[i], 
    #                   label=col_lab[i], color=col_lst[i], alpha=0.8)
    # ax[2][j].set_xlabel(r'Treatment value $t$')
    # ax[2][0].set_ylabel(r'Coverage rates for $\theta(t)$')
    # ax[2][j].set_ylim([0.56, 1])
    
    j += 1
fig.align_ylabels()
fig.tight_layout()
fig.subplots_adjust(wspace=0.2, hspace=0.11)
fig.savefig('../figures/theta.pdf')
# }}}
