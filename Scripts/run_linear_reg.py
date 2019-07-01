import sys
import pathlib

from pymc_models import hs_regression
from pymc_utils import run_model

if __name__ == "__main__":
    datapath = sys.argv[1]
    log_path = pathlib.Path.cwd() / '.logs'
    log_path.mkdir(exist_ok=True)
    logger.add(log_path / "linreg_{time}.log")
    #try:
    run_model(hs_regression, datapath=datapath)
    #except FileNotFoundError:
    #    logger.error(f'Missing file{log_path})
    logger.info("done!")
