from sklearn.externals.joblib import parallel_backend
from Spsa import Spsa


class Safs:
	def __init__(self, x, y, model):
		self.x = x
		self.y = y
		self.model = model
		self.results = None

	def run(self, num_features_selected=0, num_cores=1):
		kernel = Spsa(model=self.model, num_features_selected=num_features_selected, num_cores=num_cores)
		#kernel.load_data(data=self.data, y=self.y)
		kernel.enter_data(x=self.x, y=self.y)
		kernel.shuffle()
		kernel.initializer()
		kernel.cv_task_gen()
		with parallel_backend('multiprocessing'):
			kernel.spsa_kernel()
		self.results = kernel.parse_results()

		return self
