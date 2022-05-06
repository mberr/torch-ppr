# -*- coding: utf-8 -*-

"""The public API."""

import logging
from typing import Optional

import torch

from .utils import (
    DeviceHint,
    batched_personalized_page_rank,
    power_iteration,
    prepare_page_rank_adjacency,
    prepare_x0,
    resolve_device,
    validate_x,
)

__all__ = [
    "page_rank",
]

logger = logging.getLogger(__name__)


def page_rank(
    adj: Optional[torch.Tensor] = None,
    edge_index: Optional[torch.LongTensor] = None,
    max_iter: int = 1_000,
    alpha: float = 0.05,
    epsilon: float = 1.0e-04,
    x0: Optional[torch.Tensor] = None,
    use_tqdm: bool = False,
    device: DeviceHint = None,
) -> torch.Tensor:
    """
    Compute page rank by power iteration.

    :param adj:
        the adjacency matrix, cf. :func:`prepare_page_rank_adjacency`. Preferred over `edge_index`.
    :param edge_index: shape: `(2, m)`
        the edge index of the graph, i.e, the edge list.
    :param max_iter: $max_iter > 0$
        the maximum number of iterations
    :param alpha: $0 < alpha < 1$
        the smoothing value / teleport probability
    :param epsilon: $epsilon > 0$
        a (small) constant to check for convergence
    :param x0: shape: `(n,)`
        the initial value for $x$. If `None`, set to a constant $1/n$ vector.
    :param use_tqdm:
        whether to use a tqdm progress bar
    :param device:
        the device to use, or a hint thereof

    :return: shape: `(n,)` or `(n, batch_size)`
        the page-rank vector, i.e., a score between 0 and 1 for each node.
    """
    # normalize inputs
    adj = prepare_page_rank_adjacency(adj=adj, edge_index=edge_index)
    x0 = prepare_x0(x0=x0, n=adj.shape[0])

    # input normalization
    validate_x(x=x0, n=adj.shape[0])

    # power iteration
    return power_iteration(
        adj=adj,
        x0=x0,
        alpha=alpha,
        max_iter=max_iter,
        use_tqdm=use_tqdm,
        epsilon=epsilon,
        device=device,
    )


def personalized_page_rank(
    adj: Optional[torch.Tensor] = None,
    edge_index: Optional[torch.LongTensor] = None,
    indices: Optional[torch.Tensor] = None,
    device: DeviceHint = None,
    batch_size: Optional[int] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Personalized Page-Rank (PPR) computation.

    .. note::
        this method supports automatic memory optimization / batch size selection using :mod:`torch_max_mem`.

    :param adj: shape: (n, n)
        the adjacency matrix, cf. :func:`prepare_page_rank_adjacency`
    :param edge_index: shape: (2, m)
        the edge index, cf. :func:`prepare_page_rank_adjacency`
    :param indices: shape: (k,)
        the node indices for which to calculate the PPR. Defaults to all nodes.
    :param device:
        the device to use
    :param batch_size: >0
        the batch size. Defaults to the number of indices. It will be reduced if necessary.
    :param kwargs:
        additional keyword-based parameters passed to :func:`batched_personalized_page_rank`

    :return: shape: `(n, k)`
        the PPR vectors for each node index
    """
    # prepare adjacency and indices only once
    adj = prepare_page_rank_adjacency(adj=adj, edge_index=edge_index)
    indices = torch.arange(adj.shape[0], device=device)
    # normalize inputs
    batch_size = batch_size or len(indices)
    device = resolve_device(device=device)
    return batched_personalized_page_rank(
        adj=adj, indices=indices, device=device, batch_size=batch_size, **kwargs
    )
