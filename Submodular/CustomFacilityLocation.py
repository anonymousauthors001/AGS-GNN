import numpy

from apricot import BaseGraphSelection
from apricot.optimizers import LazyGreedy
from apricot.optimizers import ApproximateLazyGreedy
from apricot.optimizers import SieveGreedy

from tqdm import tqdm

from numba import njit
from numba import prange

dtypes = 'void(float64[:,:], float64[:], float64[:], int64[:])'
sdtypes = 'void(float64[:], int32[:], int32[:], float64[:], float64[:], int64[:])'
sieve_dtypes = 'void(float64[:,:], int64, float64[:,:], int64[:,:],' \
	'float64[:,:], float64[:], float64[:], int64[:], int64[:])' 

def calculate_gains(dtypes, parallel, fastmath, cache):
	@njit(dtypes, parallel=parallel, fastmath=fastmath, cache=cache)
	def calculate_gains_(X, gains, current_values, idxs):
		for i in prange(idxs.shape[0]):
			idx = idxs[i]
			gains[i] = numpy.maximum(X[idx], current_values).sum()
	return calculate_gains_


def calculate_gains_sparse(dtypes, parallel, fastmath, cache):
	@njit(dtypes, parallel=parallel, fastmath=fastmath, cache=cache)
	def calculate_gains_sparse_(X_data, X_indices, X_indptr, gains, current_values, idxs):
		for i in prange(idxs.shape[0]):
			idx = idxs[i]

			start = X_indptr[idx]
			end = X_indptr[idx+1]

			for j in range(start, end):
				k = X_indices[j]
				gains[i] += max(X_data[j], current_values[k]) - current_values[k]
	return calculate_gains_sparse_


def calculate_gains_sieve(dtypes, parallel, fastmath, cache):
	@njit(dtypes, parallel=parallel, fastmath=fastmath, cache=cache)
	def calculate_gains_sieve_(X, k, current_values, selections, gains, 
		total_gains, max_values, n_selected, idxs):
		n, d = X.shape
		t = max_values.shape[0]

		for j in prange(t):
			for i in range(n):
				if n_selected[j] == k:
					break

				idx = idxs[i]
				threshold = (max_values[j] / 2. - total_gains[j]) / (k - n_selected[j])
				maximum = numpy.maximum(X[i][:d], current_values[j][:d])
				gain = maximum.mean() - total_gains[j]

				if gain > threshold:
					current_values[j][:d] = maximum
					total_gains[j] = maximum.mean()

					selections[j, n_selected[j]] = idx
					gains[j, n_selected[j]] = gain
					n_selected[j] += 1

	return calculate_gains_sieve_


class FacilityLocationSelection(BaseGraphSelection):

	def __init__(self, n_samples, metric='euclidean', 
		initial_subset=None, optimizer='lazy', optimizer_kwds={}, 
		n_neighbors=None, reservoir=None, max_reservoir_size=1000,
		n_jobs=1, random_state=None, verbose=False):

		super(FacilityLocationSelection, self).__init__(n_samples=n_samples, 
			metric=metric, initial_subset=initial_subset, optimizer=optimizer, 
			optimizer_kwds=optimizer_kwds, n_neighbors=n_neighbors, 
			reservoir=reservoir, max_reservoir_size=max_reservoir_size,
			n_jobs=n_jobs, random_state=random_state, verbose=verbose)


	def fit(self, X, y=None, sample_weight=None, sample_cost=None):

		return super(FacilityLocationSelection, self).fit(X, y=y, 
			sample_weight=sample_weight, sample_cost=sample_cost)


	def _initialize(self, X_pairwise):
		super(FacilityLocationSelection, self)._initialize(X_pairwise)

		if self.initial_subset is None:
			pass
		elif self.initial_subset.ndim == 2:
			raise ValueError("When using facility location, the initial subset"\
				" must be a one dimensional array of indices.")
		elif self.initial_subset.ndim == 1:
			if not self.sparse:
				for i in self.initial_subset:
					self.current_values = numpy.maximum(X_pairwise[i],
						self.current_values).astype('float64')
			else:
				for i in self.initial_subset:
					self.current_values = numpy.maximum(
						X_pairwise[i].toarray()[0], self.current_values).astype('float64')
		else:
			raise ValueError("The initial subset must be either a two dimensional" \
				" matrix of examples or a one dimensional mask.")

		self.current_values_sum = self.current_values.sum()
		self.calculate_gains_ = calculate_gains_sparse if self.sparse else calculate_gains
		dtypes_ = sdtypes if self.sparse else dtypes

		if self.optimizer in (LazyGreedy, ApproximateLazyGreedy):
			self.calculate_gains_ = self.calculate_gains_(dtypes_, False, True, False)
		elif self.optimizer in ('lazy', 'approximate-lazy'):
			self.calculate_gains_ = self.calculate_gains_(dtypes_, False, True, False)
		else: 
			self.calculate_gains_ = self.calculate_gains_(dtypes_, True, True, False)

		#calculate_sieve_gains_ = calculate_gains_sieve_sparse if self.sparse else calculate_gains_sieve
		#dtypes_ = sieve_sparse_dtypes if self.sparse else sieve_dtypes 
		#self.calculate_sieve_gains_ = calculate_sieve_gains_(dtypes_, 
		#	True, True, False)

# 		self.calculate_sieve_gains_ = calculate_gains_sieve(sieve_dtypes,
# 			True, True, False);
#         print(X_pairwise)

	def _calculate_gains(self, X_pairwise, idxs=None):
		idxs = idxs if idxs is not None else self.idxs
		gains = numpy.zeros(idxs.shape[0], dtype='float64')

		if self.sparse:
			self.calculate_gains_(X_pairwise.data, X_pairwise.indices, 
				X_pairwise.indptr, gains, self.current_values, idxs)
		else:
			self.calculate_gains_(X_pairwise, gains, self.current_values, idxs)
			gains -= self.current_values_sum

		return gains

	def _calculate_sieve_gains(self, X_pairwise, thresholds, idxs):

		super(FacilityLocationSelection, self)._calculate_sieve_gains(
			X_pairwise,thresholds, idxs)

		if self.sparse:
			self.calculate_sieve_gains_(X_pairwise.data, X_pairwise.indices, 
				X_pairwise.indptr, self.n_samples, self.sieve_current_values_,
				self.sieve_selections_, self.sieve_gains_, 
				self.sieve_total_gains_, thresholds, 
				self.sieve_n_selected_, idxs)
		else:
			self.calculate_sieve_gains_(X_pairwise, self.n_samples, 
				self.sieve_current_values_, self.sieve_selections_, 
				self.sieve_gains_, self.sieve_total_gains_, thresholds, 
				self.sieve_n_selected_, idxs)

	def _select_next(self, X_pairwise, gain, idx):

		if self.sparse:
			self.current_values = numpy.maximum(
				X_pairwise.toarray()[0], self.current_values)
		else:
			self.current_values = numpy.maximum(X_pairwise, 
				self.current_values)

		self.current_values_sum = self.current_values.sum()

		super(FacilityLocationSelection, self)._select_next(
			X_pairwise, gain, idx)
        

if __name__ == '__main__':  
    import time
    
    X = numpy.exp(numpy.random.randn(1000, 100))
    model = FacilityLocationSelection(999, optimizer='approximate-lazy')
    start = time.time()
    model.fit(X)
    print("Time:", time.time()-start)