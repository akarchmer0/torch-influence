import logging
from typing import Callable, Optional

import numpy as np
import scipy.sparse
import scipy.sparse.linalg as L
from scipy.sparse.linalg import spsolve
import torch
from torch import nn
from torch.utils import data

from torch_influence.base import BaseInfluenceModule, BaseObjective


class AutogradInfluenceModule(BaseInfluenceModule):
    r"""An influence module that computes inverse-Hessian vector products
    by directly forming and inverting the risk Hessian matrix using :mod:`torch.autograd`
    utilities.

    Args:
        model: the model of interest.
        objective: an implementation of :class:`BaseObjective`.
        train_loader: a training dataset loader.
        test_loader: a test dataset loader.
        device: the device on which operations are performed.
        damp: the damping strength :math:`\lambda`. Influence functions assume that the
            risk Hessian :math:`\mathbf{H}` is positive definite, which often fails to
            hold for neural networks. Hence, a damped risk Hessian :math:`\mathbf{H} + \lambda\mathbf{I}`
            is used instead, for some sufficiently large :math:`\lambda > 0` and
            identity matrix :math:`\mathbf{I}`.
        check_eigvals: if ``True``, this initializer checks that the damped risk Hessian
            is positive definite, and raises a :mod:`ValueError` if it is not. Otherwise,
            no check is performed.
        mask: binary mask for Hessian entries.
        sparse: whether to use sparse routines.

    Warnings:
        This module scales poorly with the number of model parameters :math:`d`. In
        general, computing the Hessian matrix takes :math:`\mathcal{O}(nd^2)` time and
        inverting it takes :math:`\mathcal{O}(d^3)` time, where :math:`n` is the size
        of the training dataset.
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
            mask: Optional[torch.Tensor] = None,    # New parameter: binary mask for Hessian entries.
            sparse: bool = False                     # New parameter: whether to use sparse routines.
    ):
        super().__init__(
            model=model,
            objective=objective,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
        )
        self.damp = damp
        self.mask = mask
        self.sparse = sparse

        params = self._model_make_functional()
        flat_params = self._flatten_params_like(params)

        d = flat_params.shape[0]
        hess = 0.0

        for batch, batch_size in self._loader_wrapper(train=True):
            def f(theta_):
                self._model_reinsert_params(self._reshape_like_params(theta_))
                return self.objective.train_loss(self.model, theta_, batch)

            hess_batch = torch.autograd.functional.hessian(f, flat_params).detach()
            hess = hess + hess_batch * batch_size

        with torch.no_grad():
            self._model_reinsert_params(self._reshape_like_params(flat_params), register=True)
            hess = hess / len(self.train_loader.dataset)
            hess = hess + damp * torch.eye(d, device=hess.device)

            # Apply binary mask: entries set to 1 in the mask become 0 in the Hessian.
            if self.mask is not None:
                hess = hess.masked_fill(self.mask.bool(), 0)

            if check_eigvals:
                eigvals = np.linalg.eigvalsh(hess.cpu().numpy())
                logging.info("hessian min eigval %f", np.min(eigvals).item())
                logging.info("hessian max eigval %f", np.max(eigvals).item())
                if not bool(np.all(eigvals >= 0)):
                    raise ValueError()

            if self.sparse:
                # Convert dense Hessian to a SciPy sparse matrix.
                self.hess_sp = scipy.sparse.csr_matrix(hess.cpu().numpy())
                self.inverse_hess = None  # (Not precomputed in sparse mode)
            else:
                self.inverse_hess = torch.inverse(hess)

    def inverse_hvp(self, vec):
        # If using sparse mode, solve the system using a sparse solver.
        if self.sparse:
            
            vec_np = vec.cpu().numpy()
            sol = spsolve(self.hess_sp, vec_np)
            return torch.tensor(sol, device=self.device, dtype=vec.dtype)
        # Otherwise, use the computed dense inverse.
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
