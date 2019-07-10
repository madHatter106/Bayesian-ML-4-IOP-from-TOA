import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as pl
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class Ordered(pm.distributions.transforms.ElemwiseTransform):
    name = "ordered"

    def forward(self, x):
        out = tt.zeros(x.shape)
        out = tt.inc_subtensor(out[0], x[0])
        out = tt.inc_subtensor(out[1:], tt.log(x[1:] - x[:-1]))
        return out

    def forward_val(self, x, point=None):
        x, = pm.distributions.distribution.draw_values([x], point=point)
        return self.forward(x)

    def backward(self, y):
        out = tt.zeros(y.shape)
        out = tt.inc_subtensor(out[0], y[0])
        out = tt.inc_subtensor(out[1:], tt.exp(y[1:]))
        return tt.cumsum(out)

    def jacobian_det(self, y):
        return tt.sum(y[1:])


class PyMCModel:
    def __init__(self, model, X, y, **model_kws):
        self.model = model(X, y, **model_kws)

    def fit(self, n_samples=2000, **sample_kws):
        with self.model:
            self.trace_ = pm.sample(n_samples, **sample_kws)

    def fit_ADVI(self, n_samples=2000, n_iter=100000, inference='advi', **fit_kws):
        with self.model:
            self.approx_fit = pm.fit(n=n_iter, method=inference, **fit_kws)
            self.trace_ = self.approx_fit.sample(draws=n_samples)

    def show_model(self, save=False, view=True, cleanup=True):
        model_graph = pm.model_to_graphviz(self.model)
        if save:
            model_graph.render(save, view=view, cleanup=cleanup)
        if view:
            return model_graph

    def predict(self, **ppc_kws):
        ppc_ = pm.sample_ppc(self.trace_, model=self.model, **ppc_kws)
        return ppc_

    def get_waic(self):
        return pm.waic(trace=self.trace_, model=self.model)

    def get_loo(self):
        return pm.loo(trace=self.trace_, model=self.model)

    def evaluate_fit(self, show_feats):
        return pm.traceplot(self.trace_, varnames=show_feats)

    def show_forest(self, show_feats, feat_labels=None):
        g = pm.forestplot(self.trace_, varnames=show_feats,
                             ylabels=feat_labels)
        f = pl.gcf()
        try:
            ax = f.get_axes()[1]
        except IndexError:
            ax = f.get_axes()[0]
        ax.grid(axis='y')
        return g


    def plot_model_ppc_stats(self, ppc, y_obs, alpha_level1=0.05,
                             alpha_level2=0.5, ax=None):
        if ax is None:
            _, ax = pl.subplots()
        iy = np.argsort(y_obs)
        ix = np.arange(iy.size)
        ppc_mean = ppc.mean(axis=0)
        ax.scatter(ix, y_obs.values[iy], label='observed', edgecolor='k', s=50,
                   color='steelblue')
        ax.scatter(ix, ppc_mean[iy], label='prediction mean', edgecolor='k', s=50,
                   color='red')

        if alpha_level2:
            lik_hpd_2 = pm.hpd(ppc, alpha=alpha_level2)
            ax.fill_between(ix, y1=lik_hpd_2[iy, 0], y2=lik_hpd_2[iy, 1], alpha=0.5,
                            color='k',
                            label=f'prediction {1-alpha_level2:.2f}%CI',)
        if alpha_level1:
            lik_hpd_1 = pm.hpd(ppc, alpha=alpha_level1)
            ax.fill_between(ix, y1=lik_hpd_1[iy, 0], y2=lik_hpd_1[iy, 1], alpha=0.5,
                            color='k', label=f'prediction {1-alpha_level1:.2f}%CI',)
        ax.legend(loc='best')
        return ax

    def plot_model_fits2(self, y_obs, y_pred=None, title=None, ax=None, ci=0.95):
        if y_pred is None:
            y_pred = self.trace_.get_values('mu')
        y_obs = y_obs.values
        mask = np.logical_not(np.isnan(y_obs))
        y_obs = y_obs[mask]
        y_pred_mean = np.mean(y_pred, axis=0)[mask]
        y_pred_hpd = pm.hpd(y_pred, alpha=1-ci)[mask]
        xi = np.arange(y_obs.size)
        iy = np.argsort(y_obs)
        if ax is None:
            _, ax = pl.subplots(figsize=(12, 8),)
        ax.set_title(title)
        ax.plot(xi, y_obs[iy], marker='.', ls='',
                markeredgecolor='darkblue', markersize=13,
                label='observed')
        ax.plot(xi, y_pred_mean[iy], marker='o', color='indigo',
                ls='', markeredgecolor='k', alpha=0.5, label='predicted avg.')
        ax.fill_between(xi, y_pred_hpd[iy, 0], y_pred_hpd[iy, 1],
                        color='k', alpha=0.5,
                        label=f'{ci*100}%CI on pred.' );
        ax.legend(loc='best')
        return ax


def bayes_nn_model_ARD_1HL_halfCauchy_hyperpriors(X, y_obs,
                                                  n_hidden=None,
                                                  lklhd_name='likelihood'):
    """
    Hierarchical Bayesian NN Implementation with ARD and half-Cauchy
    hyper priors.

    Inputs:
    --------
    X: theano shared variable,
    y_obs: numpy vector,
    n_hidden: number of hidden layer neurons,
    lklhd_name: name of the likelihood variable.

    Output:
    --------
    PyMC3 model
    """

    if hasattr(X, 'name'):
        num_obs, num_feats = X.eval().shape
    else:
        num_obs, num_feats = X.shape

    if n_hidden is None:
        n_hidden = num_feats
    testval_bias = np.sort(np.random.randn(n_hidden))
    testval_wi1 = np.sort(np.random.randn(num_feats, n_hidden))
    with pm.Model() as model:

        hyp_wi1_sd = pm.HalfCauchy('hyp_w_i_1_sd', beta=.1, shape=(num_feats,1))
        hyp_w1o_sd = pm.HalfCauchy('hyp_w_1_out_sd', beta=.1)
        hyp_bias_1_sd = pm.HalfCauchy('hyp_bias_1_sd', beta=.1)

        w_i_1_intrmd = pm.Normal('wts_i_1_intrmd', mu=0, sd=1, shape=(num_feats, n_hidden),
                                transform=pm.distributions.transforms.Ordered(),
                                testval=testval_wi1,
                                )
        w_1_o_intrmd = pm.Normal('wts_1_o_intrmd', mu=0, sd=1, shape=(n_hidden),
                                #transform=pm.distributions.transforms.Ordered(),
                                #testval=testvals,
                                )
        w_i_1 = pm.Deterministic('wts_i_1', w_i_1_intrmd * hyp_wi1_sd)
        w_1_o = pm.Deterministic('wts_1_out', w_1_o_intrmd * hyp_w1o_sd)
        b_1 = pm.Normal('bias_1', mu=0, sd=hyp_bias_1_sd, shape=(n_hidden),
                       transform=pm.distributions.transforms.Ordered(), testval=testval_bias)
        b_o = pm.Normal('bias_o', mu=0, sd=1)
        lyr1_act = pm.math.theano.tensor.nnet.elu(tt.dot(X, w_i_1) + b_1)
        out_act = tt.dot(lyr1_act, w_1_o) + b_o

        sd = pm.HalfCauchy('sd', beta=1)
        output = pm.Normal(lklhd_name, mu=out_act, sd=sd, observed=y_obs)
        model.name = 'bnn_ARD_1HL_hC_hyp'
    return model


def bayes_nn_model_ARD_1HL_halfNormal_hyperpriors(X, y_obs,
                                             n_hidden=None,
                                             lklhd_name='likelihood'):
    """
    Hierarchical Bayesian NN Implementation with ARD.

    Inputs:
    --------
    X: theano shared variable,
    y_obs: numpy vector,
    n_hidden: number of hidden layer neurons,
    lklhd_name: name of the likelihood variable.

    Output:
    --------
    PyMC3 model
    """

    X = pm.floatX(X)
    Y = pm.floatX(y_obs)
    if hasattr(X, 'name'):
        num_obs, num_feats = X.eval().shape
    else:
        num_obs, num_feats = X.shape

    if n_hidden is None:
        n_hidden = num_feats
    testval_bias = np.sort(np.random.randn(n_hidden))
    testval_wi1 = np.sort(np.random.randn(num_feats, n_hidden))
    with pm.Model() as model:

        hyp_wi1_sd = pm.HalfNormal('hyp_w_i_1_sd', sd=.1, shape=(num_feats,1))
        hyp_w1o_sd = pm.HalfNormal('hyp_w_1_out_sd', sd=.1)
        hyp_bias_1_sd = pm.HalfNormal('hyp_bias_1_sd', sd=.1)

        w_i_1_intrmd = pm.Normal('wts_i_1_intrmd', mu=0, sd=1, shape=(num_feats, n_hidden),
                                transform=pm.distributions.transforms.Ordered(),
                                testval=testval_wi1,
                                )
        w_1_o_intrmd = pm.Normal('wts_1_o_intrmd', mu=0, sd=1, shape=(n_hidden),
                                #transform=pm.distributions.transforms.Ordered(),
                                #testval=testvals,
                                )

        w_i_1 = pm.Deterministic('wts_i_1', w_i_1_intrmd * hyp_wi1_sd)
        w_1_o = pm.Deterministic('wts_1_out', w_1_o_intrmd * hyp_w1o_sd)
        b_1 = pm.Normal('bias_1', mu=0, sd=hyp_bias_1_sd, shape=(n_hidden),
                       transform=pm.distributions.transforms.Ordered(), testval=testval_bias)
        b_o = pm.Normal('bias_o', mu=0, sd=1)
        lyr1_act = pm.Deterministic('layer1_act', pm.math.theano.tensor.nnet.elu(tt.dot(X, w_i_1) + b_1) )
        out_act = pm.Deterministic('out_act', tt.dot(lyr1_act, w_1_o) + b_o)

        sd = pm.HalfCauchy('sd', beta=1)
        output = pm.Normal(lklhd_name, mu=out_act, sd=sd, observed=y_obs)
        model.name = 'bnn_ARD_1HL_hN_hyp'
    return model


def reg_hs_regression(X, y_obs, ylabel='likelihood', **kwargs):
    """See Piironen & Vehtari, 2017 (DOI: 10.1214/17-EJS1337SI)"""
    n_features = X_.eval().shape[1]
    if tau_0 is None:
        m0 = n_features/2
        n_obs = X_.eval().shape[0]
        tau_0 = m0 / ((n_features - m0) * np.sqrt(n_obs))
    with pm.Model() as model:
        tau = pm.HalfCauchy('tau', tau_0)
        sd_bias = pm.HalfCauchy('sd_bias', beta=2.5)
        lamb_m = pm.HalfCauchy('lambda_m', beta=1)
        slab_scale = kwargs.pop('slab_scale', 3)
        slab_scale_sq = slab_scale ** 2
        slab_df = kwargs.pop('slab_df', 8)
        half_slab_df = slab_df / 2
        # Regularization bit
        c_sq = pm.InverseGamma('c_sq', alpha=half_slab_df,
                               beta=half_slab_df * slab_scale_sq)
        lamb_m_bar = tt.sqrt(c_sq) * lamb_m / (tt.sqrt(c_sq +
                                                       tt.pow(tau, 2) *
                                                       tt.pow(lamb_m, 2)
                                                      )
                                              )
        w = pm.Normal('w', mu=0, sd=tau*lamb_m_bar, shape=n_features)
        bias = pm.Laplace('bias', mu=0, b=sd_bias)
        mu_ = tt.dot(X_, w) + bias
        sig = pm.HalfCauchy('sigma', beta=5)
        y = pm.Normal(ylabel, mu=mu_, sd=sig, observed=y_obs)
        model.name = "regularized_hshoe_reg"

def hs_regression(X_, y_obs, ylabel='likelihood', tau_0=None, regularized=False, **kwargs):
    """See Piironen & Vehtari, 2017 (DOI: 10.1214/17-EJS1337SI)"""

    n_features = X_.eval().shape[1]
    if tau_0 is None:
        m0 = n_features/2
        n_obs = X_.eval().shape[0]
        tau_0 = m0 / ((n_features - m0) * np.sqrt(n_obs))
    with pm.Model() as model:
        tau = pm.HalfCauchy('tau', tau_0)
        sd_bias = pm.HalfCauchy('sd_bias', beta=2.5)
        lamb_m = pm.HalfCauchy('lambda_m', beta=1)
        w = pm.Normal('w', mu=0, sd=tau*lamb_m, shape=n_features)
        bias = pm.Laplace('bias', mu=0, b=sd_bias)
        mu_ = tt.dot(X_, w) + bias
        sig = pm.HalfCauchy('sigma', beta=5)
        y = pm.Normal(ylabel, mu=mu_, sd=sig, observed=y_obs)
        model.name = "horseshoe_reg"
    return model


def lasso_regression(X, y_obs, ylabel='y'):
    num_obs, num_feats = X.eval().shape
    with pm.Model() as mlasso:
        sd_beta = pm.HalfCauchy('sd_beta', beta=2.5)
        sig = pm.HalfCauchy('sigma', beta=2.5)
        bias = pm.Laplace('bias', mu=0, b=sd_beta)
        w = pm.Laplace('w', mu=0, b=sd_beta, shape=num_feats)
        mu_ = pm.Deterministic('mu', bias + tt.dot(X, w))
        y = pm.Normal(ylabel, mu=mu_, sd=sig, observed=y_obs.squeeze())
    return mlasso


def lasso_regr_impute_y(X, y_obs, ylabel='y'):
    num_obs, num_feats = X.eval().shape
    with pm.Model() as mlass_y_na:
        sd_beta = pm.HalfCauchy('sd_beta', beta=2.5)
        sig = pm.HalfCauchy('sigma', beta=2.5)
        bias = pm.Laplace('bias', mu=0, b=sd_beta)
        w = pm.Laplace('w', mu=0, b=sd_beta, shape=num_feats)
        mu_ = pm.Deterministic('mu', bias + tt.dot(X, w))
        mu_y_obs = pm.Normal('mu_y_obs', 0.5, 1)
        sigma_y_obs = pm.HalfCauchy('sigma_y_obs', 1)
        y_obs_ = pm.Normal('y_obs', mu_y_obs, sigma_y_obs, observed=y_obs.squeeze())
        y = pm.Normal(ylabel, mu=y_obs_, sd=sig)
    return mlass_y_na


def hier_lasso_regr(X, y_obs, add_bias=True, ylabel='y'):
    X_ = pm.floatX(X)
    Y_ = pm.floatX(y_obs)
    n_features = X_.eval().shape[1]
    with pm.Model() as mlasso:
        hyp_beta = pm.HalfCauchy('hyp_beta', beta=2.5)
        hyp_mu = pm.HalfCauchy('hyp_mu', mu=0, beta=2.5)
        sig = pm.HalfCauchy('sigma', beta=2.5)
        bias = pm.Laplace('bias', mu=hyp_mu, b=hyp_beta)
        w = pm.Laplace('w', mu=hyp_mu, b=hyp_beta, shape=n_features)
        mu_ = pm.Deterministic('mu', bias + tt.dot(X_, w))
        y = pm.Normal(ylabel, mu=mu_, sd=sig, observed=Y_)
    return mlasso


def partial_pooling_lasso(X, y_obs, ylabel='y'):
    pass
