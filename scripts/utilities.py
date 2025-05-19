import pandas as pd
import numpy as np

### Here lie the fucntions which are used both in cross_val.ipynb and in full_sample.ipynb

def to_long_format(input_df):

    df = pd.melt(input_df.reset_index(), id_vars=['index'], var_name='i', value_name='ret')
    df = df.rename(columns={'index': 't'})
    df = df.replace([np.inf, -np.inf], np.nan)

    lower = df['ret'].quantile(0.01)
    upper = df['ret'].quantile(0.99)
    df['ret'] = df['ret'].clip(lower, upper) # Truncate extreme values

    df["const"] = 1.0

    df = df.set_index(["i", "t"])
    df = df.dropna()


    return df

def estimate_k_sigma(df, resid_col='resid', entity_col='i', time_col='t', max_iter=10, tol=1e-6, silent = False):
    df = df.copy()
    df['resid_sq'] = df[resid_col] ** 2

    k_i = df.groupby(entity_col)['resid_sq'].mean()  # initial guess
    sigma_t2 = df.groupby(time_col)['resid_sq'].mean()

    for j in range(max_iter):
        k_i_old = k_i.copy()
        sigma_t2_old = sigma_t2.copy()

        # Map current estimates
        df['_k'] = df[entity_col].map(k_i)
        df['_sigma'] = df[time_col].map(sigma_t2)

        # Update sigma_t^2
        sigma_t2 = (
            df.assign(ratio=lambda x: x['resid_sq'] / x['_k'])
              .groupby(time_col)['ratio']
              .mean()
        )

        # Update k_i
        df['_sigma'] = df[time_col].map(sigma_t2)
        k_i = (
            df.assign(ratio=lambda x: x['resid_sq'] / x['_sigma'])
              .groupby(entity_col)['ratio']
              .mean()
        )

        # Convergence check
        if max((k_i - k_i_old).abs().max(), (sigma_t2 - sigma_t2_old).abs().max()) < tol:
            if silent == False: print(f"Converged after {j + 1} iterations.")
            break

    return k_i, sigma_t2