from Spsa import Spsa


class Safs:
	def __init__(self, x, y, model):
		self.x = x
		self.y = y
		self.model = model
		self.results = None

	def run(self):
		kernel = Spsa(model=self.model)
		#kernel.load_data(data=self.data, y=self.y)
		kernel.enter_data(x=self.x, y=self.y)
		kernel.shuffle()
		kernel.initializer()
		kernel.cv_task_gen()
		kernel.spsa_kernel()
		self.results = kernel.parse_results()

		return self
