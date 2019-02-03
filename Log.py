import logging


class Log:
	# create logger
	logger = logging.getLogger('SP-FSR')
	logger.setLevel(logging.INFO)

	# create console handler and set level to debug
	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)

	# create formatter
	formatter = logging.Formatter(
		fmt='[{asctime}] {name}.{levelname}: {message}',
		datefmt='%Y-%m-%d %H:%M:%S',
		style='{'
	)

	# add formatter to ch
	ch.setFormatter(formatter)

	# add ch to logger
	logger.addHandler(ch)
