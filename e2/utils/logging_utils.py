import logging


def get_logger(name: str, logging_path: str, level: str = logging.INFO) -> logging.Logger:
    """Returns a logger object.

    Args:
        name: Name of the logger.
        logging_path: Path to save the logs.
        level: Level of the logger.

    Returns:
        Logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Create file handler
    file_handler = logging.FileHandler(logging_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
