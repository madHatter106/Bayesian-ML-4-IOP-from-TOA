o"""Runs ARD BNN"""

import pickle
from datetime import datetime as DT
from loguru import logger
from theano import shared
from pymc_models import PyMCModel
from pymc_models import bayes_nn_model_ARD_1HL_halfCauchy_hyperpriors


def run_model():
    # load datasets
    with open('../PickleJar/AphiTrainTestSplitDataSets.pkl', 'rb') as fb:
        datadict = pickle.load(fb)
    X_s_train = datadict['x_train_s']
    y_train = datadict['y_train']
    X_s_test = datadict['x_test_s']
    y_test = datadict['y_test']

    bands = ct = dict.fromkeys(bands)
    model_dict== DT.now()
    time_stamp = 
    # create theano shared variable
    X_shared = shared(X_s_train.values)
    y_shared = shared(y_train.log10_aphy411.values)
    # Fitting aphi411 model:
    # Instantiate PyMC3 model with bnn likelihood
    for band in bands:
        logger.info("processing aphi{band}", band=band)
        X_shared.set_value(X_s_train.values)
        y_shared.set_value(y_train['log10_aphy%d' % band].values)
        bnn_ = PyMCModel(bayes_nn_model_ARD_1HL_halfCauchy_hyperpriors,
                            X_shared, y_train['log10_aphy%d' %band], n_hidden=4)
        bnn_.model.name = 'bnn_HL4_%d' %band
        bnn_.fit(n_samples=2000, cores=4, chains=4, tune=10000,
                    nuts_kwargs=dict(target_accept=0.95))
        ppc_train_ = bnn_.predict(likelihood_name='likelihood')
        waic_train = bnn_.get_waic()
        loo_tain = bnn_.get_loo()
        model = deepcopy(hshoe_.model)
        trace = deepcopy(hshoe_.trace_)
        run_dict = dict(model=model, trace=trace,
                        ppc_train=ppc_train_, loo_train=loo_train, waic_train=waic_train)
        X_shared.set_value(X_s_test.values)
        y_shared.set_value(y_test['log10_aphy%d' %band].values)
        ppc_test_ = bnn_.predict(likelihood_name='likelihood')
        waic_test_ = bnn.get_waic()
        loo_test = bnn.get_loo()
        run_dict.update(dict(ppc_test=ppc_test_, waic_test=waic_test, loo_test=loo_test))
        model_dict[band]=run_dict
        with open('../PickleJar/Results/bnn_model_dict_%s.pkl' %time_stamp, 'wb') as fb:
            pickle.dump(model_dict, fb, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    logger.add("linreg_{time}.log")
    run_model()
    logger.info("done!")
