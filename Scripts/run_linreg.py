import pickle
from datetime import datetime as DT
from loguru import logger
import pandas as pd
from theano import shared
from pymc_models import PyMCModel
from pymc_models import hs_regression

if __name__ == "__main__":
    logger.add("linreg_wi_{time}.log")
    # load datasets
    with open('./pickleJar/AphiTrainTestSplitDataSets.pkl', 'rb') as fb:
        datadict = pickle.load(fb)
    X_s_train = datadict['x_train_s']
    y_train = datadict['y_train']
    X_s_test = datadict['x_test_s']
    y_test = datadict['y_test']

    bands = [411, 443, 489, 510, 555, 670]
    # create band-keyed dictionary to contain models
    model_dict=dict.fromkeys(bands)

    # create theano shared variable
    X_shared = shared(X_s_train.values)

    # Fitting aphi411 model:
    # Instantiate PyMC3 model with bnn likelihood
    for band in bands:
        logger.info("processing aphi{band}", band=band)
        # set shared variable to training set
        X_shared.set_value(X_s_train.values)
        hshoe_ = PyMCModel(hs_regression,
                            X_shared, y_train['log10_aphy%d' %band])
        hshoe_.model.name = 'hshoe_aphy%d' %band
        hshoe_.fit(n_samples=2000, cores=4, chains=4, tune=10000,
                    nuts_kwargs=dict(target_accept=0.95))
        ppc_train_ = hshoe_.predict(likelihood_name='likelihood')
        # set shared variable to testing set
        X_shared.set_value(X_s_test.values)
        ppc_test_ = hshoe_.predict(likelihood_name='likelihood')
        run_dict = dict(model=hshoe_.model, trace=hshoe_.trace_,
                        ppc_train=ppc_train_, ppc_test=ppc_test_)
        model_dict[band] = run_dict
        with open('./pickleJar/Results/hshoe_model_dict_%s.pkl' %DT.now(), 'wb') as fb:
            pickle.dump(model_dict, fb, protocol=pickle.HIGHEST_PROTOCOL)
