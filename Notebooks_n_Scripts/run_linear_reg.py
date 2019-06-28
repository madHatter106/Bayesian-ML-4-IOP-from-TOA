from pymc_models import hs_regression
from pymc_utils import run_model
import pathlib

if __name__ == "__main__":

    log_path = pathlib.Path.cwd() / '.logs'
    log_path.mkdir(exist_ok=True)
    logger.add(log_path / "linreg_{time}.log")
    run_model(hs_regression)
    logger.info("done!")
