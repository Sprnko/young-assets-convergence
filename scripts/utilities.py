import numpy as np
import pandas as pd
from typing import Any, Literal, Iterable
from scipy.stats import spearmanr, pearsonr, kurtosis
from statsmodels.stats.dist_dependence_measures import distance_correlation
from tqdm import tqdm
from scipy.special import gamma
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from scipy.stats import chi2
from sklearn.model_selection import TimeSeriesSplit

import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)


# Define different functions for time decay

def K_exp(t, b): 
    """
    Exponential decay function.
    """
    return 1 + np.exp(-b * t)

def K_inv(t, b):  # ***
    """
    Inverse decay function.
    """
    return 1 + b/t

def K_power_shift(t, a, b): 
    """
    Inverse decay function.
    """
    return 1 + 1 / ((a + t)**b)

def K_log_extended(t, a, b): # ***
    """
    Extended Logarithmic decay function.
    """
    return 1 + a / np.log(100 + b * t)

def K_log(t, b): 
    """
    Extended Logarithmic decay function.
    """
    return 1 + 1 / np.log(100 + b * t)

def K_inv_power(t, b): 
    """
    Inverse power decay function.
    """
    return 1 + 1 / (t**b)

def K_const(t): 
    """
    Constant kernel function.
    """
    return 1

def K_inv_log(t, b):

    return 1 + b / np.log(100 + t)

def neg_l_gen(params, K, R, t):
    """
    Generalized Normal negative log-likelihood function.
    
    Parameters:
    - params: Model parameters (mu, alpha, beta, and K parameters).
    - K: decay function.
    - R: Returns data.
    - t: Time indices.
    
    Returns:
    - Negative log-likelihood value.
    """
    mu, alpha, beta = params[:3]
    K_params = params[3:]

    u_t = (R - K(t, *K_params) * mu) / (K(t, *K_params) * alpha)
    # u_t = (R - K(t, *K_params) * mu) / (alpha)
    nll = -np.sum(
        (np.log(beta / (2 * alpha * K(t, *K_params) * gamma(1/beta))) - np.abs(u_t) ** beta)
    )
    return nll

def neg_l_norm(params, K, R, t):
    """
    Normal negative log-likelihood function.
    
    Parameters:
    - params: Model parameters (mu, alpha, and K parameters).
    - K: decay function.
    - R: Returns data.
    - t: Time indices.
    
    Returns:
    - Negative log-likelihood value.
    """
    mu, alpha = params[:2]
    K_params = params[2:]

    u_t = R - K(t, *K_params) * mu
    
    nll = 0.5 * np.sum(np.log(2 * np.pi * (alpha ** 2) * (K(t, *K_params) ** 2)) + (u_t ** 2) / (alpha ** 2 * K(t, *K_params) ** 2))
    return nll

def OLS(params, K, R, t):

    mu = params[:1]
    K_params = params[1:]

    pred = K(t, *K_params) * mu

    e = np.sum((R - pred)**2)

    return e

def log_SR(t, a, b):

    return a + b * np.log(t)

class Returns_decay():
    
    def __init__(self, R : np.ndarray | pd.Series | list | Any, K = None, func = None, bounds_basic : Iterable = None, bounds_extra : Iterable = None):

        """
        Initialize the Returns_decay class.
        
        Parameters:
        - R: A list, NumPy array, or Pandas Series of returns.
        - K: Decay function.
        - func: Objective function
        - bounds_basic: Bounds for objective function (only for basic distribution) in the form Iterable[(min, max)]
        - bounds_extra: Bounds for objective function (additional with regard to the K function) in the form Iterable[(min, max)]
        """
        
        self.R = np.array(R)
        self.t = np.arange(1, len(R)+1)
        self.opt_model_res = None
        self.opt_const_res = None
        self.K = K
        self.func = func
        self.bounds_basic = bounds_basic
        self.bounds_extra = bounds_extra
        self.lr = None
        self.pval = None

    def settings(self, K = None, func = None, bounds_basic : Iterable = None, bounds_extra : Iterable = None):

        """
        Change the settings of the model.
        
        Parameters:
        - R: A list, NumPy array, or Pandas Series of returns.
        - K: Decay function.
        - func: Objective function
        - bounds_basic: Bounds for objective function (only for basic distribution) in the form Iterable[(min, max)]
        - bounds_extra: Bounds for objective function (additional with regard to the K function) in the form Iterable[(min, max)]
        """

        if K is not None: self.K = K
        if func is not None: self.func = func
        if bounds_basic is not None: self.bounds_basic = bounds_basic
        if bounds_extra is not None: self.K = bounds_extra

    def __repr__(self):

        """
        Representation of the Returns_decay class.
        """

        return f'Returns decay model \n num observations: {len(self.t)} \n objective function: {self.func} \n K: {self.K} \n bounds (basic and extra): {self.bounds_basic} | {self.bounds_extra}'

    def time_corr(self, type_ : Literal['pearson', 'spearman', 'distance']):

        """
        Compute time correlation using Pearson, Spearman, or distance correlation.
        
        Parameters:
        - type_: Correlation type ('pearson', 'spearman', or 'distance').
        
        Returns:
        - Correlation statistic and p-value.
        """
        
        if type_ == 'pearson' : 
            res = pearsonr(self.t, self.R)
            return res.statistic, res.pvalue
        
        elif type_ == 'spearman':
            res = spearmanr(self.t, self.R, axis=None)
            return res.statistic, res.pvalue
        
        elif type_ == 'distance':
            observed_dcor = distance_correlation(self.t, self.R)  # Compute observed dCor
            permuted_dcors = []

            for _ in tqdm(range(100)):
                R_perm = np.random.permutation(self.R)  # Shuffle Y to break dependency
                permuted_dcor = distance_correlation(self.t, R_perm)
                permuted_dcors.append(permuted_dcor)

            # Compute the p-value
            permuted_dcors = np.array(permuted_dcors)
            p_value = np.mean(permuted_dcors >= observed_dcor)

            return observed_dcor, p_value
        
    def _opt_general(self, R, t, K, bounds, population_size = 30, use_local = False):

        """
        Internal function to optimize parameters.
        
        Parameters:
        - R: Returns data
        - t: Time index
        - population_size: number of agents for differential evolution minimization.
        - K: Decay function.
        - bounds: Bounds for objective function in the form Iterable[(min, max)]
        - use_local: Whether to use a local optimizer.
        
        Returns:
        - Optimization result (global and possibly local).
        """

        # use a global optimizer
        result_global = differential_evolution(self.func, bounds=bounds, args=(K, R, t), popsize = population_size)

        if use_local:

            # use a local optimizer
            result_local = minimize(self.func, bounds=bounds, args=(K, R, t), x0 = result_global.x)
            return result_local
        
        return result_global
    
    def opt_model(self, population_size = 30, use_local = False):

        """
        Optimize model parameters.
        
        Parameters:
        - population_size: number of agents for differential evolution minimization.
        - use_local: Whether to use a local optimizer after global optimization.
        
        Returns:
        - Optimization result.
        """

        if None in [self.K, self.func, self.bounds_basic, self.bounds_extra]: raise ValueError('Model is not specified')

        bounds = self.bounds_basic + self.bounds_extra

        res = self._opt_general(self.R, self.t, self.K, bounds, population_size, use_local)
        self.opt_model_res = res

        return res
    
    def opt_const(self, population_size = 30, use_local = False):

        """
        Optimize parameters using a constant decay function.
        
        Parameters:
        - population_size: number of agents for differential evolution minimization.
        - use_local: Whether to use a local optimizer after global optimization.
        
        Returns:
        - Optimization result for the constant K model.
        """

        if None in [self.K, self.func, self.bounds_basic]: raise ValueError('Constant Model is not specified')

        res = self._opt_general(self.R, self.t, lambda t : 1, self.bounds_basic, population_size, use_local)
        self.opt_const_res = res

        return res
    
    def cross_val(self, n_splits = 5, population_size = 30, use_local = False):

        if None in [self.K, self.func, self.bounds_basic, self.bounds_extra]: raise ValueError('Model is not specified')

        tscv = TimeSeriesSplit(n_splits = n_splits)    
        bounds = self.bounds_basic + self.bounds_extra
        df = len(self.bounds_extra)

        res = pd.DataFrame(columns=['LR', 'pval']).rename_axis('Fold #')

        for i, (train_index, test_index) in enumerate(tscv.split(self.R)):
            print(f"Fitting Fold #{i}")

            train_R, train_t = self.R[train_index], self.t[train_index]
            test_R, test_t = self.R[test_index], self.t[test_index]

            model_result = self._opt_general(train_R, train_t, self.K, bounds, population_size, use_local)
            model_nll_test = self.func(model_result.x, self.K, test_R, test_t)

            constant_result = self._opt_general(train_R, train_t, lambda t : 1, self.bounds_basic, population_size, use_local)
            constant_nll_test = self.func(constant_result.x, lambda t : 1, test_R, test_t)

            lr = 2 * (constant_nll_test - model_nll_test)
            pval = 1 - chi2.cdf(lr, df)

            res.loc[i,:] = [lr, pval]

        return res



    def LR_test(self, model_result = None, constant_result = None):

        """
        Perform a likelihood ratio test between the model and constant model.
        
        Parameters:
        - model_result: Optimized result for the model (optional, defaults to best available result).
        - constant_result: Optimized result for the constant model (optional, defaults to best available result).
        
        Returns:
        - Likelihood ratio test statistic.
        - p-value of the test.
        """

        lr, pval = self._LR_test(model_result = model_result, constant_result = constant_result)

        self.lr = lr
        self.pval = pval

        return lr, pval
                
    def _LR_test(self, model_result = None, constant_result = None):

        """
        Internal : Perform a likelihood ratio test between the model and constant model.
        
        Parameters:
        - model_result: Optimized result for the model (optional, defaults to best available result).
        - constant_result: Optimized result for the constant model (optional, defaults to best available result).
        
        Returns:
        - Likelihood ratio test statistic.
        - p-value of the test.
        """

        if model_result is None:

            if self.opt_model_res is None: raise ValueError("Model hasn't been fit yet, use opt_params")
            else: model_result = self.opt_model_res

        if constant_result is None:

            if self.opt_const_res is None: raise ValueError("Constant model hasn't been fit yet, use const_params")
            else: constant_result = self.opt_const_res

        lr = 2 * (constant_result.fun - model_result.fun)
        df = len(model_result.x) - len(constant_result.x)
        pval = 1 - chi2.cdf(lr, df = df)

        return lr, pval
        


if __name__ == '__main__':

    ### Example Usage

    import yfinance as yf
    import datetime

    TICKER = 'BTC-USD'
    START = '2000-01-01'
    END = datetime.datetime.today().date()
    TF = '1d'

    data = yf.download(TICKER, period='max', interval=TF, progress=False)['Adj Close']
    # data_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
    # data = data.reindex(data_range)
    # data = data.ffill()
    data = data.pct_change(1) * 100  
    data = data.dropna()

    RD = Returns_decay(data, K = K_inv, func = neg_l_gen, bounds_basic = [(1e-6, 1), (1e-6, 100), (1e-6, 100)], bounds_extra = [(1e-6, 100)])

    res = RD.cross_val(population_size=1000, n_splits=5)
    print(res)


    # RD.opt_model(population_size=1000)
    # RD.opt_const(population_size=1000)

    # print(f'model x: {RD.opt_model_res.x}\nconstant x: {RD.opt_const_res.x}')
    # print(f'model func: {RD.opt_model_res.fun}\nconstant func: {RD.opt_const_res.fun}')
    # print(RD.LR_test()) 