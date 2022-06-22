import os, logging


def init_logging(log_file=None, level='INFO', mode='w'):
    handlers = [logging.StreamHandler()]
    format_str = '[%(asctime)s] %(message)s'
    datefmt_str = '%Y-%m-%d %H:%M:%S'
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode=mode))
    if level == 'INFO':
        logging.basicConfig(handlers=handlers, format=format_str, datefmt=datefmt_str,
                            level=logging.INFO)
    elif level == 'DEBUG':
        logging.basicConfig(handlers=handlers, format=format_str, datefmt=datefmt_str,
                            level=logging.DEBUG)
