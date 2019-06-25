from pymc_models import hs_regression
from pymc_utils import run_model


if __name__ == "__main__":
    logger.add("linreg_{time}.log")
    run_model(hs_regression)
    logger.info("done!")
