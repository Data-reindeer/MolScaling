import math
import torch
from time import time
import numpy as np
from tqdm import tqdm
import pdb

import torch


def _kpp(data: torch.Tensor, k: int, sample_size: int = -1):
    """ Picks k points in the data based on the kmeans++ method.
    Parameters
    ----------
    data : torch.Tensor
        Expect a rank 1 or 2 array. Rank 1 is assumed to describe 1-D
        data, rank 2 multidimensional data, in which case one
        row is one observation.
    k : int
        Number of samples to generate.
    sample_size : int
        sample data to avoid memory overflow during calculation
    Returns
    -------
    init : ndarray
        A 'k' by 'N' containing the initial centroids.
    References
    ----------
    .. [1] D. Arthur and S. Vassilvitskii, "k-means++: the advantages of
       careful seeding", Proceedings of the Eighteenth Annual ACM-SIAM Symposium
       on Discrete Algorithms, 2007.
    .. [2] scipy/cluster/vq.py: _kpp
    """
    if sample_size > 0:
        data = data[torch.randint(0, int(data.shape[0]),
                                  [min(100000, data.shape[0])], device=data.device)]
    dims = data.shape[1] if len(data.shape) > 1 else 1
    init = torch.zeros((k, dims)).to(data.device)

    r = torch.distributions.uniform.Uniform(0, 1)
    for i in range(k):
        if i == 0:
            init[i, :] = data[torch.randint(data.shape[0], [1])]

        else:
            D2 = torch.cdist(init[:i, :][None, :], data[None, :], p=2)[0].amin(dim=0)
            probs = D2 / torch.sum(D2)
            cumprobs = torch.cumsum(probs, dim=0)
            init[i, :] = data[torch.searchsorted(
                cumprobs, r.sample([1]).to(data.device))]
    return init


def _krandinit(data: torch.Tensor, k: int, sample_size: int = -1):
    """Returns k samples of a random variable whose parameters depend on data.
    More precisely, it returns k observations sampled from a Gaussian random
    variable whose mean and covariances are the ones estimated from the data.
    Parameters
    ----------
    data : torch.Tensor
        Expect a rank 1 or 2 array. Rank 1 is assumed to describe 1-D
        data, rank 2 multidimensional data, in which case one
        row is one observation.
    k : int
        Number of samples to generate.
    sample_size : int
        sample data to avoid memory overflow during calculation
    Returns
    -------
    x : ndarray
        A 'k' by 'N' containing the initial centroids
    References
    ----------
    .. [1] scipy/cluster/vq.py: _krandinit
    """
    mu = data.mean(axis=0)
    if sample_size > 0:
        data = data[torch.randint(0, int(data.shape[0]),
                                  [min(100000, data.shape[0])], device=data.device)]
    if data.ndim == 1:
        cov = torch.cov(data)
        x = torch.randn(k, device=data.device)
        x *= np.sqrt(cov)
    elif data.shape[1] > data.shape[0]:
        # initialize when the covariance matrix is rank deficient
        _, s, vh = data.svd(data - mu, full_matrices=False)
        x = torch.randn(k, s.shape[0])
        sVh = s[:, None] * vh / torch.sqrt(data.shape[0] - 1)
        x = x.dot(sVh)
    else:
        cov = torch.atleast_2d(torch.cov(data.T))

        # k rows, d cols (one row = one obs)
        # Generate k sample of a random variable ~ Gaussian(mu, cov)
        x = torch.randn(k, mu.shape[0], device=data.device)
        x = torch.matmul(x, torch.linalg.cholesky(cov).T)
    x += mu
    return x


def _kpoints(data, k, sample_size=-1):
    """Pick k points at random in data (one row = one observation).
    Parameters
    ----------
    data : ndarray
        Expect a rank 1 or 2 array. Rank 1 are assumed to describe one
        dimensional data, rank 2 multidimensional data, in which case one
        row is one observation.
    k : int
        Number of samples to generate.
    sample_size : int (not used)
        sample data to avoid memory overflow during calculation
    Returns
    -------
    x : ndarray
        A 'k' by 'N' containing the initial centroids
    """
    return data[torch.randint(0, data.shape[0], size=[k])]


init_methods = {
    "gaussian": _krandinit,
    "kmeans++": _kpp,
    "random": _kpoints,
}

class KMeans:
  '''
  Kmeans clustering algorithm implemented with PyTorch
  Parameters:
    n_clusters: int, 
      Number of clusters
    max_iter: int, default: 100
      Maximum number of iterations
    tol: float, default: 0.0001
      Tolerance
    
    verbose: int, default: 0
      Verbosity
    mode: {'euclidean', 'cosine'}, default: 'euclidean'
      Type of distance measure
      
    init_method: {'random', 'point', '++'}
      Type of initialization
    minibatch: {None, int}, default: None
      Batch size of MinibatchKmeans algorithm
      if None perform full KMeans algorithm
      
  Attributes:
    centroids: torch.Tensor, shape: [n_clusters, n_features]
      cluster centroids
  '''
  def __init__(self, n_clusters, max_iter=100, tol=0.0001, verbose=0, mode="cosine", 
               minibatch=None, init_method="random", device='cpu', centroids=None):
    self.n_clusters = n_clusters
    self.max_iter = max_iter
    self.tol = tol
    self.verbose = verbose
    self.mode = mode
    self.init_method = init_method
    self.minibatch = minibatch
    self._loop = False
    self._show = False
    self.device = device

    try:
      import PYNVML
      self._pynvml_exist = True
    except ModuleNotFoundError:
      self._pynvml_exist = False
    
    self.centroids = centroids

  @staticmethod
  def cos_sim(a, b):
    """
      Compute cosine similarity of 2 sets of vectors
      Parameters:
      a: torch.Tensor, shape: [m, n_features]
      b: torch.Tensor, shape: [n, n_features]
    """
    a_norm = a.norm(dim=-1, keepdim=True)
    b_norm = b.norm(dim=-1, keepdim=True)
    a = a / (a_norm + 1e-8)
    b = b / (b_norm + 1e-8)
    return a @ b.transpose(-2, -1)

  @staticmethod
  def euc_sim(a, b):
    """
      Compute euclidean similarity of 2 sets of vectors
      Parameters:
      a: torch.Tensor, shape: [m, n_features]
      b: torch.Tensor, shape: [n, n_features]
    """
    return 2 * a @ b.transpose(-2, -1) -(a**2).sum(dim=1)[..., :, None] - (b**2).sum(dim=1)[..., None, :]
  
  @staticmethod
  def tan_sim(a, b):
    """
      Compute euclidean similarity of 2 sets of vectors
      Parameters:
      a: torch.Tensor, shape: [m, n_features]
      b: torch.Tensor, shape: [n, n_features]
    """
    # A_norm = A.sum(dim=-1, keepdim=True).repeat(1, B.shape[0])
    # B_norm = B.sum(dim=-1, keepdim=True).repeat(1, A.shape[0]).T

    A_norm = torch.square(a.norm(dim=-1, keepdim=True, p=2)).repeat(1, b.shape[0])
    B_norm = torch.square(b.norm(dim=-1, keepdim=True, p=2)).repeat(1, a.shape[0]).T

    AB = a @ b.T
    return 1 - (AB / (A_norm + B_norm - AB))

  def remaining_memory(self):
    """
      Get remaining memory in gpu
    """
    torch.cuda.synchronize(self.device)
    torch.cuda.empty_cache()
    if self._pynvml_exist:
      pynvml.nvmlInit()
      gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
      info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
      remaining = info.free
    else:
      remaining = torch.cuda.memory_allocated(self.device)
    return remaining

  def max_sim(self, a, b):
    """
      Compute maximum similarity (or minimum distance) of each vector
      in a with all of the vectors in b
      Parameters:
      a: torch.Tensor, shape: [m, n_features]
      b: torch.Tensor, shape: [n, n_features]
    """
    device = a.device.type
    batch_size = a.shape[0]
    if self.mode == 'cosine':
      sim_func = self.cos_sim
    elif self.mode == 'euclidean':
      sim_func = self.euc_sim
    elif self.mode == 'tanimoto':
      sim_func = self.tan_sim

    if device == 'cpu':
      sim = sim_func(a, b)
      max_sim_v, max_sim_i = sim.max(dim=-1)
      return max_sim_v, max_sim_i
    else:
      if a.dtype == torch.double:
        expected = a.shape[0] * a.shape[1] * b.shape[0] * 8
      if a.dtype == torch.float:
        expected = a.shape[0] * a.shape[1] * b.shape[0] * 4
      elif a.dtype == torch.half:
        expected = a.shape[0] * a.shape[1] * b.shape[0] * 2
      ratio = math.ceil(expected / self.remaining_memory())
      subbatch_size = math.ceil(batch_size / ratio)
      msv, msi = [], []
      for i in range(ratio):
        if i*subbatch_size >= batch_size:
          continue
        sub_x = a[i*subbatch_size: (i+1)*subbatch_size]
        sub_sim = sim_func(sub_x, b)
        sub_max_sim_v, sub_max_sim_i = sub_sim.max(dim=-1)
        del sub_sim
        msv.append(sub_max_sim_v)
        msi.append(sub_max_sim_i)
      if ratio == 1:
        max_sim_v, max_sim_i = msv[0], msi[0]
      else:
        max_sim_v = torch.cat(msv, dim=0)
        max_sim_i = torch.cat(msi, dim=0)
      return max_sim_v, max_sim_i

  def fit_predict(self, X, centroids=None):
    """
      Combination of fit() and predict() methods.
      This is faster than calling fit() and predict() seperately.
      Parameters:
      X: torch.Tensor, shape: [n_samples, n_features]
      centroids: {torch.Tensor, None}, default: None
        if given, centroids will be initialized with given tensor
        if None, centroids will be randomly chosen from X
      Return:
      labels: torch.Tensor, shape: [n_samples]
    """
    assert isinstance(X, torch.Tensor), "input must be torch.Tensor"
    assert X.dtype in [torch.half, torch.float, torch.double], "input must be floating point"
    assert X.ndim == 2, "input must be a 2d tensor with shape: [n_samples, n_features] "

    batch_size, emb_dim = X.shape
    device = self.device
    start_time = time()
    if centroids is None:
      self.centroids = init_methods[self.init_method](X, self.n_clusters, self.minibatch)
    else:
      self.centroids = centroids
    self.centroids = self.centroids.to(device)
    num_points_in_clusters = torch.ones(self.n_clusters, device=device, dtype=X.dtype)
    closest = None
    for i in tqdm(range(self.max_iter)):
      iter_time = time()
      if self.minibatch is not None:
        x = X[np.random.choice(batch_size, size=[self.minibatch], replace=False)]
      else:
        x = X
      x = x.to(device)
      closest = self.max_sim(a=x, b=self.centroids)[1]
      matched_clusters, counts = closest.unique(return_counts=True)

      c_grad = torch.zeros_like(self.centroids)
      expanded_closest = closest[None].expand(self.n_clusters, -1)
      mask = (expanded_closest==torch.arange(self.n_clusters, device=device)[:, None]).to(X.dtype)
      c_grad = mask @ x / mask.sum(-1)[..., :, None]
      c_grad[c_grad!=c_grad] = 0 # remove NaNs

      error = (c_grad - self.centroids).pow(2).mean()
      # print('error: {}'.format(error))
      if self.minibatch is not None:
        lr = 1/num_points_in_clusters[:,None] * 0.9 + 0.1
        # lr = 1/num_points_in_clusters[:,None]**0.1 
      else:
        lr = 1
      num_points_in_clusters[matched_clusters] += counts
      self.centroids = self.centroids * (1-lr) + c_grad * lr
      if self.verbose >= 2:
        print('iter:', i, 'error:', error.item(), 'time spent:', round(time()-iter_time, 4))
      if error <= self.tol:
        break

    if self.verbose >= 1:
      print(f'used {i+1} iterations ({round(time()-start_time, 4)}s) to cluster {batch_size} items into {self.n_clusters} clusters')
    return self.centroids

  def predict(self, X):
    """
      Predict the closest cluster each sample in X belongs to
      Parameters:
      X: torch.Tensor, shape: [n_samples, n_features]
      Return:
      labels: torch.Tensor, shape: [n_samples]
    """
    assert isinstance(X, torch.Tensor), "input must be torch.Tensor"
    assert X.dtype in [torch.half, torch.float, torch.double], "input must be floating point"
    assert X.ndim == 2, "input must be a 2d tensor with shape: [n_samples, n_features] "
    X = X.to(self.device)
    return self.max_sim(a=X, b=self.centroids)[0], self.max_sim(a=X, b=self.centroids)[1]

  def fit(self, X, centroids=None):
    """
      Perform kmeans clustering
      Parameters:
      X: torch.Tensor, shape: [n_samples, n_features]
    """
    assert isinstance(X, torch.Tensor), "input must be torch.Tensor"
    assert X.dtype in [torch.half, torch.float, torch.double], "input must be floating point"
    assert X.ndim == 2, "input must be a 2d tensor with shape: [n_samples, n_features] "

    self.fit_predict(X, centroids)