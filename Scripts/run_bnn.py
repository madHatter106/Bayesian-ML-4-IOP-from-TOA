"""Runs ARD BNN"""

import pickle
from datetime import datetime as DT
from loguru import logger
from theano import shared
from pymc_models import PyMCModel
from pymc_models import bayes_nn_model_ARD_1HL_halfCauchy_hyperpriors


if __name__ == "__main__":
    # set logger
    logger.add("bnn_{time}.log")

    # load datasets
    with open('./pickleJar/AphiTrainTestSplitDataSets.pkl', 'rb') as fb:
        datadict = pickle.load(fb)
    X_s_train = datadict['x_train_s']
    y_train = datadict['y_train']
    X_s_test = datadict['x_test_s']
    y_test = datadict['y_test']

    bands = [411, 443, 489, 510, 555, 670]
    model_dict=dict.fromkeys(bands)

    # create theano shared variable
    X_shared = shared(X_s_train.values)

    # Fitting aphi411 model:
    # Instantiate PyMC3 model with bnn likelihood
    for band in bands:
        logger.info("processing aphi{band}", band=band)
        X_shared.set_value(X_s_train.values)
        bnn_ = PyMCModel(bayes_nn_model_ARD_1HL_halfCauchy_hyperpriors,
                            X_shared, y_train['log10_aphy%d' %band], n_hidden=4)
        bnn_.model.name = 'bnn_HL4_%d' %band
        bnn_.fit(n_samples=2000, cores=4, chains=4, tune=10000,
                    nuts_kwargs=dict(target_accept=0.95))
        ppc_train_ = bnn_.predict(likelihood_name='likelihood')
        X_shared.set_value(X_s_test.values)
        ppc_test_ = bnn_.predict(likelihood_name='likelihood')
        run_dict = dict(model=bnn_.model, trace=bnn_.trace_,
                        ppc_train=ppc_train_, ppc_test=ppc_test_)
        model_dict[band]=run_dict
        with open('./pickleJar/Results/bnn_model_dict_%s.pkl' %DT.now(), 'wb') as fb:
            pickle.dump(model_dict, fb, protocol=pickle.HIGHEST_PROTOCOL)
