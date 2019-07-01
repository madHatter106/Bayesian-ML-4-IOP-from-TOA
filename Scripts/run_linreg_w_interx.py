"""Runs Linear Reg, w/ Interactions"""

import pathlib
import sys

from loguru import logger

from pymc_utils import run_model
from pymc_models import hs_regression


if __name__ == "__main__":
    datapath = sys.argv[1]
    log_path = pathlib.Path.cwd() / '.logs'
    log_path.mkdir(exist_ok=True)
    logger.add("linreg_wi_{time}.log")
    run_model(hs_regression, logger, compute_interactions=True, datapath=datapath)
    logger.info("done!")
