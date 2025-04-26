#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: Yikun Zhang
Last Editing: Dec 11, 2024

Description: Simulation 2 with non-separable error in the outcome model.
'''

import numpy as np
import torch
import normflows as nf
import larsflow as lf
from matplotlib import pyplot as plt
from tqdm import tqdm
import scipy.stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

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
    K = 4 # number of layers for the flow
    torch.manual_seed(10)

    latent_size = outcome.size()[1] # dimension of the latent space
    context_size = contexts.size()[1] # dimension of the context / contexts space
    hidden_units = 256 # hidden_units for MaskedAffineAutoregressive
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
        # a = nf.nets.MLP([latent_size, 64, 128, 256, 512, 256, 128, 1], output_fn='sigmoid') # 7 layers
        a = nf.nets.MLP([latent_size, 256, 256, 1], output_fn='sigmoid') # 4 layers
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
def train(model, outcome=None, contexts=None, max_iter=2000, sample_size=1000, lr=1e-3, weight_decay=1e-3, 
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
        b_contexts = contexts[boot_idx]
        b_outcome = outcome[boot_idx]

        # b_contexts = xn[boot_idx]
        # b_outcome = yn[boot_idx]

        loss = model.forward_kld(x=b_outcome, context=b_contexts) # loss function is defined with forward KL divergence
        # loss = model.reverse_kld(b_outcome, b_contexts) # loss function is defined with reverse KL divergence

        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward(retain_graph=True)
            optimizer.step()
            # Log loss

        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
# }}}

#=======================================================================================#

# reg_mod_lst = ['LR', 'RF', 'NN']
reg_mod_lst = ['NN']
# cond_type = ['true', 'kde', 'nf']
cond_type = ['true', 'nf']
fac_lst = [0.75, 1, 1.25, 1.5, 2]
n_lst = [500, 1000, 2000]

# Start the loop {{{
for reg in reg_mod_lst:
    # reg = 'NN'
    for fac in fac_lst:
        for cond in cond_type:
        # cond = 'true'
            for n in n_lst:
            n = 1000
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
                    T_sim_tensor = torch.tensor(T_sim).view(n, 1).to(device)
                    X_sim_tensor = torch.tensor(X_sim).to(device)
                    # set up flow model
                    flowTS = flow_model(base='resampled', outcome=T_sim_tensor, contexts=X_sim_tensor)
                    # log the loss and train the flow
                    loss_hist = np.array([])
                    train(model=flowTS, outcome=T_sim_tensor, contexts=X_sim_tensor,
                          max_iter=100000, sample_size=n)
                    # plot the loss
                    plt.plot(loss_hist, label='loss')
                    plt.show(block=False)
                    zz = T_sim_tensor
                    # context_temp = torch.tensor(0.1).to(device).repeat(zz.size()[0], 20)
                    context_temp = X_sim_tensor
                    # calculate the conditional probability for the sample
                    log_prob = flowTS.log_prob(zz, context_temp).to('cpu')
                    prob = torch.exp(log_prob)
                    prob[torch.isnan(prob)] = 0
                    nf_cond = prob.detach().numpy()
                    plt.bar(T_sim, height=nf_cond)
                    plt.show()
                    cond_mod = nf_cond
                    cond = 'true'

                    # plot the true density and estimation from nf
                    plt.bar(T_sim, true_cond)
                    plt.show(block=False)

                elif cond == 'kde':
                    # cond = 'kde'
                    # Conditional density model
                    regr_nn2 = MLPRegressor(hidden_layer_sizes=(20,), activation='relu', learning_rate='adaptive', 
                                    learning_rate_init=0.1, random_state=1, max_iter=200)
                    cond_mod = regr_nn2
                else:
                    cond_mod = cond_reg_mod
                    
                # m_est_dr5, sd_est_dr5 = DRCurve(Y=Y_sim, X=X_dat, t_eval=t_qry, est='DR', 
                #                                 mu=reg_mod, condTS_type=cond, condTS_mod=cond_mod,
                #                                 tau=0.001, L=5, h=h, kern='epanechnikov', h_cond=None,
                #                                 print_bw=False, self_norm=True)

                m_est_dr1, sd_est_dr1 = DRCurve(Y=Y_sim, X=X_dat, t_eval=t_qry, est='DR', 
                                                mu=reg_mod, condTS_type=cond, condTS_mod=cond_mod, 
                                                tau=0.001, L=1, h=h, kern='epanechnikov', h_cond=None, 
                                                print_bw=False, self_norm=True)

                # m_est_nf1, sd_est_nf1 = DRCurve(Y=Y_sim, X=X_dat, t_eval=t_qry, est='DR', 
                #                                 mu=reg_mod, condTS_type=cond, condTS_mod=cond_mod, 
                #                                 tau=0.001, L=1, h=h, kern='epanechnikov', h_cond=None, 
                #                                 print_bw=False, self_norm=True)
                #
                # m_est_kde1, sd_est_kde1 = DRCurve(Y=Y_sim, X=X_dat, t_eval=t_qry, est='DR', 
                #                                 mu=reg_mod, condTS_type=cond, condTS_mod=cond_mod, 
                #                                 tau=0.001, L=1, h=h, kern='epanechnikov', h_cond=None, 
                #                                 print_bw=False, self_norm=True)

                with open('../Results/Simulation2_m_est'+str(job_id)+'_'+str(reg)+'_h'+str(fac)+'_condmod_'+str(cond)+'_n_'+str(n)+'_selfnorm.dat', 'wb+') as file:
                    # pickle.dump([m_est_dr5, sd_est_dr5, m_est_dr1, sd_est_dr1], file)
                
                if reg == 'LR':
                    reg_mod = LinearRegression()
                elif reg == 'RF':
                    reg_mod = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=1)
                elif reg == 'NN':
                    reg_mod = NeurNet
                
                # theta_dr5, theta_sd5 = DRDerivCurve(Y=Y_sim, X=X_dat, t_eval=t_qry, est='DR', 
                #                                             beta_mod=reg_mod, n_iter=1000, 
                #                                   lr=0.01, condTS_type=cond, condTS_mod=cond_mod, 
                #                                   tau=0.001, L=5, h=h, kern='epanechnikov', 
                #                                   h_cond=None, print_bw=True, delta=0.01, self_norm=True)
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

reg = 'NN'
fac_lst = [0.75, 1, 1.25, 1.5, 2, 4]
cond_type = ['true', 'nf']
t_qry = np.linspace(-2, 2, 81)
m_true = 1.2*t_qry + t_qry**2
n = 2000

model_lst = ['nn']

for fac in fac_lst:
    fac = 2
    for cond in cond_type:
        cond = 'true'
        with open('./Results/Simulation2_theta_est'+str(job_id)+'_'+str(reg)+'_h'+str(fac)+'_condmod_'+str(cond)+'_n_'+str(n)+'_selfnorm.dat', 'rb+') as file:
        # with open('../Results/Simulation2_m_est_'+str(reg)+'_h'+str(fac)+'_condmod_'+str(cond)+'_n_'+str(n)+'_nobnd.dat', "rb") as file:
            m_dr1_lst, sd_dr1_lst = pickle.load(file)
        
        # Bias
        # m_ra1_bias = Bias(np.array(m_ra1_lst), m_true)
        #
        # m_ipw1_bias = Bias(np.array(m_ipw1_lst), m_true)
        
        m_dr1_bias = Bias(np.array(m_dr1_lst), m_true)
        
        # RMSE
        # m_ra1_rmse = RMSE(np.array(m_ra1_lst), m_true)
        
        # m_ipw1_rmse = RMSE(np.array(m_ipw1_lst), m_true)
        
        m_dr1_rmse = RMSE(np.array(m_dr1_lst), m_true)
        
        # Coverage
        m_dr1_cov = CovProb(np.array(m_dr1_lst), np.array(sd_dr1_lst), m_true, alpha=0.95)
        
        if cond == 'true':
            res_bias = 
            np.column_stack([t_qry, m_dr1_bias])
            #
            # res_rmse = np.column_stack([t_qry, m_ra1_rmse, m_ipw1_rmse, m_dr1_rmse])
            res_cov = np.column_stack([t_qry, m_dr1_cov])
        else:
            res_bias = np.column_stack([res_bias, m_dr1_bias])
            res_rmse = np.column_stack([res_rmse, m_dr1_rmse])
            res_cov = np.column_stack([res_cov, m_dr1_cov])
                                           
    
    res_bias1 = pd.DataFrame(res_bias)
    res_bias1.columns = ['Query point', 'DR (true, L=1)', 'DR (nf, L=1)']
    
    res_rmse1 = pd.DataFrame(res_rmse)
    res_rmse1.columns = ['Query point', 'DR (true, L=1)', 'DR (nf, L=1)']
    
    res_cov1 = pd.DataFrame(res_cov)
    res_cov1.columns = ['Query point', 'DR (true, L=1)', 'DR (nf, L=1)']
    
    
    # plt.figure(figsize=(6,4))
    for i in range(res_bias1.shape[1] - 1):
        plt.plot(t_qry, res_bias1.iloc[:,i+1], label=res_bias1.columns[i+1])
    plt.ylim([-2, 2])
    plt.xlabel('Query points')
    plt.ylabel(r'Bias for $m(t)$')
    plt.title(r'$h=$'+str(fac)+'$\hat{\sigma}_T n^{-1/5}$')
    plt.legend(bbox_to_anchor=(1, 1))
plt.show()
