import logging

LOGGING_FORMAT = "%(levelname)s --- %(name)s --- %(message)s"

def configure_logging():
    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)