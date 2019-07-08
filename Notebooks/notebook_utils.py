import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

import matplotlib.pyplot as pl
from matplotlib import rcParams
from seaborn import heatmap
from cmocean import cm as cmo

def plot_pca_res(pca_machine, threshold=0.85, alpha=1,
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

        
def plot_cross_corr(df_pca, df, ax=None, **heatmap_kws):
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
    

def fit_plotter(x, y, fit_fn=None, transform=None, noise=None):
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

    
def create_smry(trc, labels, vname=['w']):
    ''' Conv fn: create trace summary for sorted forestplot '''
    dfsm = pm.summary(trc, varnames=vname)
    dfsm.rename(index={wi: lbl for wi, lbl in zip(dfsm.index, feature_labels)},
                inplace=True)
    #dfsm.sort_values('mean', ascending=True, inplace=True)
    dfsm['ypos'] = np.linspace(1, 0, len(dfsm))
    return dfsm


def custom_forestplot(df, ax, replace_bathy=True):
    ax.scatter(x=df['mean'], y=df.ypos, edgecolor='k', facecolor='white', zorder=2)
    ax.hlines(df.ypos, xmax=df['hpd_97.5'], xmin=df['hpd_2.5'],
              color='k', zorder=1, linewidth=3)
    ax.set_yticks(df.ypos)
    ax.set_yticklabels(df.index.tolist())
    ax.axvline(linestyle=':', color='k')
    ax.grid(axis='y', zorder=0)

    
def plot_pairwise_map(df, ax=None, annot=False):
    if ax is None:
        _, ax = pl.subplots(figsize=(20, 20))
    dfc = df.corr().iloc[1:, :-1]
    heatmap(dfc, vmin=-1, vmax=1, cmap=cmo.balance_r, annot=annot, annot_kws={'fontsize': 6},
            ax=ax, mask=np.triu(np.ones([dfc.shape[1]]*2), k=1), fmt='.1f',
           linewidths=0.5, linecolor='black')
    ax.set_facecolor('k')
    return ax


def plot_obs_against_ppc(y_obs, ppc, ax=None, plot_1_to_1=False,
                         add_label=True, **scatter_kwds):
    if ax is None:
        _, ax = pl.subplots(figsize=(10, 10))
    ppc_mean = ppc.mean(axis=0)
    mae = mean_absolute_error(y_obs, ppc_mean)
    r2 = r2_score(y_obs, ppc_mean)
    if add_label:
        scatter_lbl = scatter_kwds.pop('label', '')
        scatter_lbl = fr'{scatter_lbl}; {r2:.2f}; {mae:.2f}'
        ax.scatter(y_obs, ppc_mean, edgecolor='k', label=scatter_lbl, **scatter_kwds)
    else:
        ax.scatter(y_obs, ppc_mean, edgecolor='k', **scatter_kwds)
    if plot_1_to_1:
        min_ = min(ppc_mean.min(), y_obs.min())
        max_ = max(ppc_mean.max(), y_obs.max())
        ax.plot([min_, max_], [min_, max_], ls='--', color='k', label='1:1')
    ax.legend(loc='upper left')
    return ax


def plot_fits_w_estimates(y_obs, ppc, ax=None, legend=False):
    """ Plot Fits with Uncertainty Estimates"""
    iy  = np.argsort(y_obs)
    ix = np.arange(iy.size)
    lik_mean = ppc.mean(axis=0)
    lik_hpd = pm.hpd(ppc)
    lik_hpd_05 = pm.hpd(ppc, alpha=0.5)
    r2 = r2_score(y_obs, lik_mean)
    mae = mean_absolute_error(y_obs, lik_mean)
    if ax is None:
        _, ax = pl.subplots(figsize=(12, 8))
    ax.scatter(ix, y_obs.values[iy], label='observed', edgecolor='k', s=40,
               color='w', marker='d', zorder=2);
    ax.scatter(ix, lik_mean[iy], label='model mean -- $r^2$=%.2f -- mae=%.2f' %(r2, mae),
               edgecolor='k', s=40, color='w', zorder=3)

    ax.fill_between(ix, y1=lik_hpd_05[iy, 0], y2=lik_hpd_05[iy, 1], color='gray', 
                   label='model output 50%CI', zorder=1,linestyle='-', lw=2, edgecolor='k');
    ax.fill_between(ix, y1=lik_hpd[iy, 0], y2=lik_hpd[iy, 1], color='k', alpha=0.75,
                   label='model output 95%CI', zorder=0, );
    if legend:
        ax.legend(loc='upper left');
    return ax


def compute_fig_height(fig_width):
    golden_mean = (sqrt(5)-1.0)/2.0    # Aesthetic ratio
    return fig_width*golden_mean # height in inches


def latexify(fig_width=None, fig_height=None, columns=1, square=False):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    PNAS 1-column figure width should be 3.5"
    PNAS 2-column wide figures should be 4.49" or 7" (??)
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples
    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf
    
    assert(columns in [1,2])

    if fig_width is None:
        fig_width = 3.43 if columns==1 else 4.49 # width in inches

    if fig_height is None:
        if square:
            fig_height = fig_width
        else:
            fig_height =  compute_fig_height(fig_width) # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height + 
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'ps',
              'text.latex.preamble': [r'\usepackage{gensymb}'],
              'axes.labelsize': 8, # fontsize for x and y labels (was 10)
              'axes.titlesize': 8,
              'font.size': 8, # was 10
              'legend.fontsize': 8, # was 10
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'text.usetex': True,
              'figure.figsize': [fig_width,fig_height],
              'font.family': 'serif'
             }

    rcParams.update(params)