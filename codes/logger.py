import logging


class Logger:
    def __init__(self):
        logging.basicConfig(level=logging.INFO,
                            filename='output.log',
                            datefmt='%Y/%m/%d %H:%M:%S',
                            format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
        self.logger = logging.getLogger(
            __name__)
        self.file_handler = logging.FileHandler(
            'output.log')
        self.file_handler.setLevel(
            level=logging.INFO)
        self.formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.file_handler.setFormatter(
            self.formatter)
        self.logger.addHandler(self.file_handler)

    def getLogger(self):
        return self.logger
