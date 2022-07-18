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
    validate_adjacency,
    validate_x,
)

__all__ = [
    "page_rank",
    "personalized_page_rank",
]

logger = logging.getLogger(__name__)


def page_rank(
    *,
    adj: Optional[torch.Tensor] = None,
    edge_index: Optional[torch.LongTensor] = None,
    num_nodes: Optional[int] = None,
    add_identity: bool = False,
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
        the adjacency matrix, cf. :func:`torch_ppr.utils.prepare_page_rank_adjacency`. Preferred over ``edge_index``.
    :param edge_index: shape: ``(2, m)``
        the edge index of the graph, i.e, the edge list. cf. :func:`torch_ppr.utils.prepare_page_rank_adjacency`
    :param num_nodes:
        the number of nodes used to determine the shape of the adjacency matrix.
        If ``None``, and ``adj`` is not already provided, it is inferred from ``edge_index``.
    :param add_identity:
        whether to add an identity matrix to ``A`` to ensure that each node has a degree of at least one.

    :param max_iter: ``max_iter > 0``
        the maximum number of iterations
    :param alpha: ``0 < alpha < 1``
        the smoothing value / teleport probability
    :param epsilon: ``epsilon > 0``
        a (small) constant to check for convergence
    :param x0: shape: ``(n,)``
        the initial value for ``x``. If ``None``, set to a constant $1/n$ vector,
        cf. :func:`torch_ppr.utils.prepare_x0`. Otherwise, the tensor is checked for being valid using
        :func:`torch_ppr.utils.validate_x`.
    :param use_tqdm:
        whether to use a tqdm progress bar
    :param device:
        the device to use, or a hint thereof


    :return: shape: ``(n,)`` or ``(batch_size, n)``
        the page-rank vector, i.e., a score between 0 and 1 for each node.
    """
    # normalize inputs
    adj = prepare_page_rank_adjacency(
        adj=adj, edge_index=edge_index, num_nodes=num_nodes, add_identity=add_identity
    )
    validate_adjacency(adj=adj)

    x0 = prepare_x0(x0=x0, n=adj.shape[0])

    # input normalization
    validate_x(x=x0, n=adj.shape[0])

    # power iteration
    x = power_iteration(
        adj=adj,
        x0=x0,
        alpha=alpha,
        max_iter=max_iter,
        use_tqdm=use_tqdm,
        epsilon=epsilon,
        device=device,
    )
    if x.ndim < 2:
        return x
    return x.t()


def personalized_page_rank(
    *,
    adj: Optional[torch.Tensor] = None,
    edge_index: Optional[torch.LongTensor] = None,
    add_identity: bool = False,
    num_nodes: Optional[int] = None,
    indices: Optional[torch.Tensor] = None,
    device: DeviceHint = None,
    batch_size: Optional[int] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Personalized Page-Rank (PPR) computation.

    .. note::
        this method supports automatic memory optimization / batch size selection using :mod:`torch_max_mem`.

    :param adj: shape: ``(n, n)``
        the adjacency matrix, cf. :func:`torch_ppr.utils.prepare_page_rank_adjacency`
    :param edge_index: shape: ``(2, m)``
        the edge index, cf. :func:`torch_ppr.utils.prepare_page_rank_adjacency`
    :param num_nodes:
        the number of nodes used to determine the shape of the adjacency matrix.
        If ``None``, and ``adj`` is not already provided, it is inferred from ``edge_index``.
    :param add_identity:
        whether to add an identity matrix to ``A`` to ensure that each node has a degree of at least one.

    :param indices: shape: ``(k,)``
        the node indices for which to calculate the PPR. Defaults to all nodes.
    :param device:
        the device to use
    :param batch_size: ``batch_size > 0``
        the batch size. Defaults to the number of indices. It will be reduced if necessary.
    :param kwargs:
        additional keyword-based parameters passed to :func:`torch_ppr.utils.batched_personalized_page_rank`

    :return: shape: ``(k, n)``
        the PPR vectors for each node index
    """
    # resolve device first
    device = resolve_device(device=device)
    # prepare adjacency and indices only once
    adj = prepare_page_rank_adjacency(
        adj=adj, edge_index=edge_index, num_nodes=num_nodes, add_identity=add_identity
    ).to(device=device)
    validate_adjacency(adj=adj)

    if indices is None:
        indices = torch.arange(adj.shape[0], device=device)
    else:
        indices = torch.as_tensor(indices, dtype=torch.long, device=device)
    # normalize inputs
    batch_size = batch_size or len(indices)
    return batched_personalized_page_rank(
        adj=adj, indices=indices, device=device, batch_size=batch_size, **kwargs
    ).t()
