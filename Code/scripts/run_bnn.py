"""Runs ARD BNN"""

import pathlib
import sys

from loguru import logger

from pymc_utils import run_model
from pymc_models import bayes_nn_model_ARD_1HL_halfCauchy_hyperpriors as bnn




if __name__ == "__main__":
    datapath = sys.argv[1]

    log_path = pathlib.Path.cwd() / '.logs'
    log_path.mkdir(exist_ok=True)
    logger.add(log_path / "bnn_{time}.log")
    run_model(bnn, logger, datapath=datapath)
    logger.info("done!")
