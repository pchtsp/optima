import pandas as pd
import pytups.superdict as sd

def normalize_variables(X, mean_std=None, orient='index'):
    if mean_std is not None:
        tab = pd.DataFrame.from_dict(mean_std, orient=orient)
        # we need to keep the order of the columns on the df
        # probably lost by the dictionary
        mean = tab['mean'].filter(X.columns)
        std = tab['std'].filter(X.columns)
    else:
        mean = X.mean()
        std = X.std()
    return X/mean
    # return (X - mean)/std
    # return (X - mean)


def get_mean_std(X):
    stds = X.std().to_dict()
    means = X.mean().to_dict()
    return {k: {'std': v, 'mean': means[k]} for k, v in stds.items()}


def denormalize(X_norm, mean_std):
    keys = list(X_norm.columns)
    mean_std_f = sd.SuperDict.from_dict(mean_std).filter(keys)
    tab = pd.DataFrame.from_dict(mean_std_f, orient='index')
    return ((X_norm * tab['std']) + tab['mean'])
