
class PDAS(object):
	"""
	PDAS (Progressive Deep Architecture Search).
	Searching promising deep architectures in a progressive way.

	At first, we construct blocks with b=1 cell, and construct deep architecture from block specifications.
	Then we train the TRUE/PROXY deep architecture on the training set, and evaluate the validation performances.
	At this time, we train a reward predictor from scratch, use it to predict performance when b=2...B.

	Then for `b = 2 to B`,
	  1. we first expand the current block with one more cells (same with previous cells in this block),
	  2. then predict performance using the reward predictor,
	  3. then select most promising (TopK) blocks according to predictions.
	  4. After got K blocks, we construct deep architecture from them each, and train them on the training set
	    and evaluate them on the validation set.
	  5. Finally, we fine-tune the reward predictor with new performance data.

	Currently, we use an MLP-ensemble model as our reward predictor, as [1].
	[1] Liu C, Zoph B, Neumann M, et al. Progressive Neural Architecture Search[J]. 2017.
	"""
	def __init__(self, all_algorithms: dict=None):
		self.all_algorithms = all_algorithms

	def run(self):
		pass


def cell_to_DeepArchitecture():
	pass
