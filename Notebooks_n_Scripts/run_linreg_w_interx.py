"""Runs ARD BNN"""



if __name__ == "__main__":
    log_path = pathlib.Path.cwd() / '.logs'
    log_path.mkdir(exist_ok=True)
    logger.add("linreg_wi_{time}.log")
    run_model(hs_regression, compute_interactions=True)
    logger.info("done!")
