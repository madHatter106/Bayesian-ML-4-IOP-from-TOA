from copy import deepcopy
from datetime import datetime as DT
import pickle
from copy import deepcopy
from loguru import logger
import pandas as pd
from theano import shared
from pymc_models import PyMCModel
from pymc_models import hs_regression

def run_model():
    with open('../PickleJar/AphiTrainTestSplitDataSets.pkl', 'rb') as fb:
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
        waic_train = hshoe_.get_waic()
        loo_train = hshoe_.get_loo()
        model = deepcopy(hshoe_.model)
        trace = deepcopy(hshoe_.trace)
        run_dict = dict(model=model, trace=trace,
                        ppc_train=ppc_train_, loo_train=loo_train, waic_train=waic_train)
        # set shared variable to testing set
        X_shared.set_value(X_s_test.values)
        ppc_test_ = hshoe_.predict(likelihood_name='likelihood')
        waic_test = hshoe_.get_waic()
        loo_test = hshoe_.get_loo()
        run_dict.update(dict(ppc_test=ppc_test_, waic_test=waic_test, loo_test=loo_test))
        model_dict[band] = run_dict
        with open('../PickleJar/Results/hshoe_model_dict_%s.pkl' %DT.now(), 'wb') as fb:
            pickle.dump(model_dict, fb, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    logger.add("linreg_{time}.log")
    run_model()
    logger.info("done!")
    # load datasets
