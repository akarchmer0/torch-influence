import logging
from typing import Callable, Optional, Sequence

import numpy as np
import scipy.sparse
import scipy.sparse.linalg as L
from scipy.sparse.linalg import spsolve
import torch
from torch import nn
from torch.utils import data

from tqdm import tqdm

from torch_influence.base import BaseInfluenceModule, BaseObjective

# --- Helper function for flattening parameters ---
def flatten_params(params):
    """Flattens a list of tensors into a single 1D tensor.
    Returns the flat tensor and a list of the original shapes.
    """
    flat_list = []
    shapes = []
    for p in params:
        shapes.append(p.shape)
        flat_list.append(p.reshape(-1))
    return torch.cat(flat_list), shapes

# Targeted Hessian computation function
def compute_targeted_hessian_efficient(model, train_loader, important_param_indices, objective_fn, device="cuda", damp=0.01, params=None, module=None):
    """Compute a Hessian submatrix for selected parameters efficiently using vectorized operations.
    
    This version computes the full Hessian for each batch but only extracts and accumulates
    the elements corresponding to important parameters.
    """
    if params is None:
        params = [p for p in model.parameters()]
    
    # Get the flat parameter vector
    flat_params = module._flatten_params_like(params)
    
    # Create a Hessian submatrix for just the important indices
    n_important = len(important_param_indices)
    targeted_hessian = torch.zeros((n_important, n_important), device=device)
    
    # Convert important_param_indices to tensor for indexing
    idx_tensor = torch.tensor(important_param_indices, device=device)
    
    n_samples = 0
    # Loop over batches and accumulate the Hessian submatrix
    for batch in train_loader:
        # Transfer batch to device
        batch = module._transfer_to_device(batch)
        batch_size = batch[0].size(0)
        n_samples += batch_size
        
        # Define the loss function
        def batch_loss_fn(params_):
            # Insert parameters into model
            module._model_reinsert_params(module._reshape_like_params(params_))
            # Forward pass
            outputs = model(batch[0])
            # Loss calculation
            return objective_fn(outputs, batch)
        
        # Compute full Hessian for this batch (using PyTorch's vectorized implementation)
        full_hess_batch = torch.autograd.functional.hessian(batch_loss_fn, flat_params)
        
        # Extract only the submatrix we care about using vectorized indexing
        # This creates a view of the tensor, so no additional memory is allocated
        batch_submatrix = full_hess_batch[idx_tensor[:, None], idx_tensor]
        
        # Accumulate weighted by batch size
        targeted_hessian += batch_submatrix * batch_size
    
    # Normalize by the total number of samples
    targeted_hessian = targeted_hessian / n_samples
    
    # Apply damping
    targeted_hessian += damp * torch.eye(n_important, device=device)
    
    return targeted_hessian

# --- Targeted AutogradInfluenceModule that integrates targeted Hessian ---
class AutogradInfluenceModule(BaseInfluenceModule):
    """An influence module that computes inverse-Hessian vector products
    by directly forming and inverting the risk Hessian matrix using torch.autograd.
    
    Added parameter:
        important_param_indices: If provided (as a list or sequence of indices), a targeted Hessian
            (i.e. a submatrix for just these parameters) is computed instead of the full Hessian.
    """

    def __init__(
            self,
            model: nn.Module,
            objective: BaseObjective,
            train_loader: data.DataLoader,
            test_loader: data.DataLoader,
            device: torch.device,
            damp: float,
            check_eigvals: bool = False,
            sparse: bool = False,                     # Whether to use sparse routines.
            important_param_indices: Optional[Sequence[int]] = None  # NEW parameter.
    ):
        super().__init__(
            model=model,
            objective=objective,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
        )
        self.damp = damp
        self.sparse = sparse
        self.important_param_indices = important_param_indices

        # Make the model functional and get a flattened parameter vector.
        params = self._model_make_functional()
        flat_params = self._flatten_params_like(params)
        
        # Decide whether to compute the full Hessian or only a targeted submatrix.
        if important_param_indices is not None:
            # Use the targeted Hessian code.
            hess = compute_targeted_hessian_efficient(
                model=self.model,
                train_loader=self.train_loader,
                important_param_indices=important_param_indices,
                objective_fn=self.objective.train_loss_on_outputs,  # Pass the method directly
                device=str(self.device),
                damp=self.damp,
                params=params,
                module=self  # Pass a reference to self
            )
            d = hess.shape[0]
        else:
            d = flat_params.shape[0]
            hess = 0.0
            for batch, batch_size in self._loader_wrapper(train=True):
                def f(theta_):
                    self._model_reinsert_params(self._reshape_like_params(theta_))
                    return self.objective.train_loss(self.model, theta_, batch)
                hess_batch = torch.autograd.functional.hessian(f, flat_params).detach()
                hess = hess + hess_batch * batch_size
            hess = hess / len(self.train_loader.dataset)
            hess = hess + damp * torch.eye(d, device=hess.device)
        
        if check_eigvals:
            eigvals = np.linalg.eigvalsh(hess.cpu().numpy())
            logging.info("hessian min eigval %f", np.min(eigvals).item())
            logging.info("hessian max eigval %f", np.max(eigvals).item())
            if not bool(np.all(eigvals >= 0)):
                raise ValueError("The damped Hessian is not positive definite.")
        
        if self.sparse:
            self.hess_sp = scipy.sparse.csr_matrix(hess.cpu().numpy())
            self.inverse_hess = None  # Not precomputed in sparse mode.
        else:
            self.inverse_hess = torch.inverse(hess)

        self._model_reinsert_params(self._reshape_like_params(flat_params), register=True)

    def inverse_hvp(self, vec):
        # If using sparse mode with important_param_indices
        if self.sparse and hasattr(self, 'important_param_indices'):
            # Extract only the components of vec that correspond to important parameters
            important_indices = torch.tensor(self.important_param_indices, device=self.device)
            vec_subset = vec[important_indices]
            
            # Solve the smaller system
            vec_np = vec_subset.cpu().numpy()
            sol_subset = spsolve(self.hess_sp, vec_np)
            
            # Create a solution vector of the original size, initialized with zeros
            full_sol = torch.zeros_like(vec)
            
            # Place the solution elements back in their original positions
            full_sol[important_indices] = torch.tensor(sol_subset, device=self.device, dtype=vec.dtype)
            
            return full_sol
        # Original implementation for non-targeted cases
        elif self.sparse:
            vec_np = vec.cpu().numpy()
            sol = spsolve(self.hess_sp, vec_np)
            return torch.tensor(sol, device=self.device, dtype=vec.dtype)
        # Otherwise, use the computed dense inverse
        return self.inverse_hess @ vec


class CGInfluenceModule(BaseInfluenceModule):
    r"""An influence module that computes inverse-Hessian vector products
    using the method of (truncated) Conjugate Gradients (CG).

    This module relies :func:`scipy.sparse.linalg.cg()` to perform CG.

    Args:
        model: the model of interest.
        objective: an implementation of :class:`BaseObjective`.
        train_loader: a training dataset loader.
        test_loader: a test dataset loader.
        device: the device on which operations are performed.
        damp: the damping strength :math:`\lambda`. Influence functions assume that the
            risk Hessian :math:`\mathbf{H}` is positive-definite, which often fails to
            hold for neural networks. Hence, a damped risk Hessian :math:`\mathbf{H} + \lambda\mathbf{I}`
            is used instead, for some sufficiently large :math:`\lambda > 0` and
            identity matrix :math:`\mathbf{I}`.
        gnh: if ``True``, the risk Hessian :math:`\mathbf{H}` is approximated with
            the Gauss-Newton Hessian, which is positive semi-definite.
            Otherwise, the risk Hessian is used.
        **kwargs: keyword arguments which are passed into the "Other Parameters" of
            :func:`scipy.sparse.linalg.cg()`.
    """

    def __init__(
            self,
            model: nn.Module,
            objective: BaseObjective,
            train_loader: data.DataLoader,
            test_loader: data.DataLoader,
            device: torch.device,
            damp: float,
            gnh: bool = False,
            **kwargs
    ):
        super().__init__(
            model=model,
            objective=objective,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
        )

        self.damp = damp
        self.gnh = gnh
        self.cg_kwargs = kwargs
        self.n_params = sum(p.numel() for p in model.parameters())
        print("Number of parameters: ", self.n_params)

    def inverse_hvp(self, vec):
        params = self._model_make_functional()
        flat_params = self._flatten_params_like(params)

        def hvp_fn(v):
            v = torch.tensor(v, requires_grad=False, device=self.device, dtype=vec.dtype)

            hvp = 0.0
            for batch, batch_size in self._loader_wrapper(train=True):
                hvp_batch = self._hvp_at_batch(batch, flat_params, vec=v, gnh=self.gnh)
                hvp = hvp + hvp_batch.detach() * batch_size
            hvp = hvp / len(self.train_loader.dataset)
            hvp = hvp + self.damp * v

            return hvp.cpu().numpy()

        d = vec.shape[0]
        linop = L.LinearOperator((d, d), matvec=hvp_fn)
        ihvp = L.cg(A=linop, b=vec.cpu().numpy(), **self.cg_kwargs)[0]

        with torch.no_grad():
            self._model_reinsert_params(self._reshape_like_params(flat_params), register=True)

        return torch.tensor(ihvp, device=self.device)


class LiSSAInfluenceModule(BaseInfluenceModule):
    r"""An influence module that computes inverse-Hessian vector products
    using the Linear time Stochastic Second-Order Algorithm (LiSSA).

    At a high level, LiSSA estimates an inverse-Hessian vector product
    by using truncated Neumann iterations:

    .. math::
        \mathbf{H}^{-1}\mathbf{v} \approx \frac{1}{R}\sum\limits{r = 1}^R
        \left(\sigma^{-1}\sum{t = 1}^T(\mathbf{I} - \sigma^{-1}\mathbf{H}{r, t})^t\mathbf{v}\right)

    Here, :math:`\mathbf{H}` is the risk Hessian matrix and :math:`\mathbf{H}{r, t}` are
    loss Hessian matrices over batches of training data drawn randomly with replacement (we
    also use a batch size in ``train_loader``). In addition, :math:`\sigma > 0` is a scaling
    factor chosen sufficiently large such that :math:`\sigma^{-1} \mathbf{H} \preceq \mathbf{I}`.

    In practice, we can compute each inner sum recursively. Starting with
    :math:`\mathbf{h}{r, 0} = \mathbf{v}`, we can iteratively update for :math:`T` steps:

    .. math::
        \mathbf{h}{r, t} = \mathbf{v} + \mathbf{h}{r, t - 1} - \sigma^{-1}\mathbf{H}{r, t}\mathbf{h}{r, t - 1}

    where :math:`\mathbf{h}{r, T}` will be equal to the :math:`r`-th inner sum.

    Args:
        model: the model of interest.
        objective: an implementation of :class:`BaseObjective`.
        train_loader: a training dataset loader.
        test_loader: a test dataset loader.
        device: the device on which operations are performed.
        damp: the damping strength :math:`\lambda`. Influence functions assume that the
            risk Hessian :math:`\mathbf{H}` is positive-definite, which often fails to
            hold for neural networks. Hence, a damped risk Hessian :math:`\mathbf{H} + \lambda\mathbf{I}`
            is used instead, for some sufficiently large :math:`\lambda > 0` and
            identity matrix :math:`\mathbf{I}`.
        repeat: the number of trials :math:`R`.
        depth: the recurrence depth :math:`T`.
        scale: the scaling factor :math:`\sigma`.
        gnh: if ``True``, the risk Hessian :math:`\mathbf{H}` is approximated with
            the Gauss-Newton Hessian, which is positive semi-definite.
            Otherwise, the risk Hessian is used.
        debug_callback: a callback function which is passed in :math:`(r, t, \mathbf{h}_{r, t})`
            at each recurrence step.
     """

    def __init__(
            self,
            model: nn.Module,
            objective: BaseObjective,
            train_loader: data.DataLoader,
            test_loader: data.DataLoader,
            device: torch.device,
            damp: float,
            repeat: int,
            depth: int,
            scale: float,
            gnh: bool = False,
            debug_callback: Optional[Callable[[int, int, torch.Tensor], None]] = None
    ):

        super().__init__(
            model=model,
            objective=objective,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
        )

        self.damp = damp
        self.gnh = gnh
        self.repeat = repeat
        self.depth = depth
        self.scale = scale
        self.debug_callback = debug_callback

    def inverse_hvp(self, vec):

        params = self._model_make_functional()
        flat_params = self._flatten_params_like(params)

        ihvp = 0.0

        for r in range(self.repeat):

            h_est = vec.clone()

            for t, (batch, _) in enumerate(self._loader_wrapper(sample_n_batches=self.depth, train=True)):

                hvp_batch = self._hvp_at_batch(batch, flat_params, vec=h_est, gnh=self.gnh)

                with torch.no_grad():
                    hvp_batch = hvp_batch + self.damp * h_est
                    h_est = vec + h_est - hvp_batch / self.scale

                if self.debug_callback is not None:
                    self.debug_callback(r, t, h_est)

            ihvp = ihvp + h_est / self.scale

        with torch.no_grad():
            self._model_reinsert_params(self._reshape_like_params(flat_params), register=True)

        return ihvp / self.repeat
