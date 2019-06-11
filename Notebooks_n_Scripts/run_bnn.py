"""Runs ARD BNN"""

from loguru import logger
from pymc_utils import run_model
from pymc_models import bayes_nn_model_ARD_1HL_halfCauchy_hyperpriors as bnn


if __name__ == "__main__":
    logger.add("linreg_{time}.log")
    run_model(bnn)
    logger.info("done!")
