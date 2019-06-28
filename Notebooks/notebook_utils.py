import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
from seaborn import heatmap
from cmocean import cm as cmo


def PlotPCARes(pca_machine, threshold=0.85, alpha=1,
               num_pca_disp=None, ax=None):
    """Plot PCA results."""
    if ax is None:
        _, ax = pl.subplots(figsize=(12, 10))
    cum_expl_var = np.cumsum(pca_machine.explained_variance_ratio_)
    if num_pca_disp is None:
        num_pca_disp = np.argmax(cum_expl_var > 0.999) + 1

    ax.bar(range(1, num_pca_disp+1),
           pca_machine.explained_variance_ratio_[:num_pca_disp],
           align='center', color='skyblue',
           label='PC explained_variance')
    ax.step(range(1, num_pca_disp+1),
            np.cumsum(pca_machine.explained_variance_ratio_[:num_pca_disp]),
            where='mid',
            label='cumulated variance')
    ax.hlines(threshold, 0, num_pca_disp+2, linestyles='--', linewidth=2,
              label='selection cutoff: %.2f' % threshold)
    ax.set_xticks(np.arange(1, num_pca_disp+1))
    ax.set_xticklabels(['PC%d' % i for i in range(1, num_pca_disp+1)],
                       rotation=45)
    ax.set_xlim((0.5, 0.5+num_pca_disp))
    ax.set_ylim((0, 1))
    ax.set_title('PCA Explained Variance')
    ax.legend(loc='center right')

        
def PlotCrossCorr(df_pca, df, ax=None, **heatmap_kws):
    dfrrs_w_pca = pd.merge(df_pca, df, 'outer',
                            left_index=True,
                            right_index=True)

    corr_w_pca = dfrrs_w_pca.corr().T
    corr_w_pca.drop(df_pca.columns, axis=0, inplace=True)
    corr_w_pca.drop(df.columns, axis=1, inplace=True)
    if ax is None:
        _, ax = pl.subplots(figsize=(20, 5))
    heatmap(corr_w_pca, cmap=cmo.balance, annot=True,
            vmin=-1, vmax=1, ax=ax, **heatmap_kws);
    

def FitPlotter(x, y, fit_fn=None, transform=None, noise=None):
    line_kwargs = {'color': 'blue'}
    dot_kwargs = {'alpha': 0.5}
    if fit_fn:
        if transform is None:
            transform = lambda j: j
        x_data = np.atleast_2d(np.linspace(x.min(), x.max(), num=500)).T
        y_fit = fit_fn(transform(x_data))
        if y_fit.ndim == 2 and y_fit.shape[1] > 1:
            n_lines = y_fit.shape[1]
            if n_lines > 100:
                indices = np.linspace(0, n_lines - 1, num=100, dtype=int)
                y_fit = y_fit[:, indices]
                n_lines = len(indices)
            line_kwargs['alpha'] = 0.3
            line_kwargs['linewidth'] = 2
            x_data = np.repeat(np.atleast_2d(x_data), n_lines, axis=1)
        pl.plot(x_data, y_fit, '-', **line_kwargs)
        if noise is not None:
            noise = noise[indices]
            noise_kwargs = line_kwargs.copy()
            noise_kwargs['color'] = 'steelblue'
            noise_kwargs['linewidth'] = line_kwargs['linewidth'] * 0.5
            for const in (-2, -1, 1, 2):
                noise_kwargs['alpha'] = 0.5 * line_kwargs['alpha'] / abs(const)
                pl.plot(x_data, y_fit + const * noise, '-', **noise_kwargs)
    plt.plot(x, y, 'ro', **dot_kwargs)
