import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt

class BayesianDBM(object):
    
    n_steps = 1000
    
    def __init__(self, observations, prior=(1, 1), alpha=0.8):
        self.observations = np.array(observations)
        
        self.fails = np.concatenate(([0], np.cumsum(~self.observations)))
        self.succeses = np.concatenate(([0], np.cumsum(self.observations)))
        
        self.prior = np.array(prior)
        self.alpha = alpha
        
        self.n_observations = len(self.observations)
        
        self.t  = np.linspace(0, 1, self.n_steps)
        
    def get_beta_distributions(self):
        
        weights = [[1.0]]

        beta_dists = [sp.stats.beta(self.prior[0], self.prior[1])]


        for obs in np.arange(1, self.n_observations):
            weights.append([(1-self.alpha)])

            alphas = self.prior[0] + self.succeses[obs] - self.succeses[obs - np.arange(obs+1)]
            betas = self.prior[1] + self.fails[obs] - self.fails[obs - np.arange(obs+1)]

            beta_dists.append(sp.stats.beta(alphas, betas))

            for age in np.arange(1, obs):
                weights[-1].append((1-self.alpha) * self.alpha**(age))

            weights[-1].append(self.alpha**(obs+1))


        return weights, beta_dists
    
    def get_p_per_trial(self):
        return np.sum(self.get_full_posterior_per_trial() * self.t[:, np.newaxis], 0)
    

            
    def get_full_posterior_per_trial(self):        
        means = []
        
        weights, betas = self.get_beta_distributions()

        for i in xrange(self.n_observations):
            means.append(np.sum(weights[i] * betas[i].pdf(self.t[:, np.newaxis]), 1) / self.n_steps)
            
        return np.array(means).T
    
    
    def plot_full_posterior_per_trial(self, **kwargs):
        
        cmap = kwargs.pop('cmap', plt.cm.coolwarm)
        
        aspect = self.n_observations / 2.
        aspect = kwargs.pop('aspect', aspect)

        extent = -.5, self.n_observations-.5, 0, 1
        
        plt.imshow(self.get_full_posterior_per_trial(), cmap=cmap, origin='lower', extent=extent, aspect=aspect, **kwargs)
        
        plt.xlim(0, self.n_observations - .5)
                
        
