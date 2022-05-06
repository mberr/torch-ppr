# -*- coding: utf-8 -*-

"""Main code."""

import logging
from typing import Collection, Optional, Union

import torch
from tqdm.auto import tqdm

__all__ = [
    "page_rank",
]

logger = logging.getLogger(__name__)

DeviceHint = Union[None, str, torch.device]


def resolve_device(device: DeviceHint) -> torch.device:
    """
    Resolve the device to use.

    :param device:
        the device hint

    :return:
        the resolved device
    """
    # pass-through torch.device
    if isinstance(device, torch.device):
        return device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    device = torch.device(device=device)
    logger.info(f"Resolved device={device}")
    return device


def edge_index_to_sparse_matrix(
    edge_index: torch.LongTensor, num_nodes: Optional[int] = None
) -> torch.Tensor:
    """
    Convert an edge index to a sparse matrix.

    :param edge_index: shape: $(2, m)$
        the edge index
    :param num_nodes:
        the number of nodes used to determine the shape of the matrix.
        If `None`, it is inferred from `edge_index`.

    :return: shape: $(n, n)$
        the adjacency matrix as a sparse tensor
    """
    if num_nodes is None:
        num_nodes = edge_index.max().values.item() + 1
        logger.info(f"Inferred num_nodes={num_nodes}")
    return torch.sparse_coo_tensor(
        indices=edge_index,
        values=torch.ones_like(edge_index[0], dtype=torch.get_default_dtype()),
        size=(num_nodes, num_nodes),
    )


def prepare_page_rank_adjacency(
    adj: Optional[torch.Tensor] = None, edge_index: Optional[torch.LongTensor] = None
) -> torch.Tensor:
    """
    Prepare the page-rank adjacency matrix.

    :param adj: shape: $(n, n)$
        the adjacency matrix
    :param edge_index: shape: $(2, m)$
        the edge index

    :raises ValueError:
        if neither is provided

    :return: shape: $(n, n)$
        the symmetric, normalized, and sparse adjacency matrix
    """
    if adj is not None:
        return adj

    if edge_index is None:
        raise ValueError("Must provide at least one of `adj` and `edge_index`.")

    # convert to sparse matrix, shape: (n, n)
    adj = edge_index_to_sparse_matrix(edge_index=edge_index)
    # symmetrize
    adj = adj + adj.t()
    # adjacency normalization
    degree_inv = torch.reciprocal(adj.sum(dim=0))
    degree_inv = torch.sparse_coo_tensor(
        indices=torch.arange(degree_inv.shape[0]).unsqueeze(dim=0).repeat(2, 1), values=degree_inv
    )
    return torch.sparse.mm(adj, degree_inv)


def power_iteration(
    adj: torch.Tensor,
    x0: torch.Tensor,
    alpha: float = 0.05,
    max_iter: int = 1_000,
    use_tqdm: bool = False,
    epsilon: float = 1.0e-04,
    device: DeviceHint = None,
) -> torch.Tensor:
    r"""
    Perform the power iteration.

    .. math::
        \mathbf{x}^{(i+1)} = (1 - \alpha) \cdot \mathbf{A} \mathbf{x}^{(i)} + \alpha \mathbf{x}^{(0)}

    :param adj: shape: $(n, n)$
        the (sparse) adjacency matrix
    :param x0: shape: $(n,)$, or $(n, batch_size)$
        the initial value for $x$.
    :param alpha: $0 < alpha < 1$
        the smoothing value / teleport probability
    :param epsilon: $epsilon > 0$
        a (small) constant to check for convergence
    :param use_tqdm:
        whether to use a tqdm progress bar

    :return: shape: $(n,)$ or $(n, batch_size)$
        the $x$ value after convergence (or maximum number of iterations).
    """
    # normalize device
    device = resolve_device(device=device)
    # send tensors to device
    adj = adj.to(device=device)
    x0 = x0.to(device=device)
    # power iteration
    x_old = x = x0
    beta = 1.0 - alpha
    progress = tqdm(range(max_iter), unit_scale=True, leave=False, disable=not use_tqdm)
    for i in progress:
        # calculate beta * A.dot(x) + alpha * x
        x = torch.sparse.addmm(mat=x0, mat1=adj, mat2=x0, beta=alpha, alpha=beta)
        max_diff = torch.linalg.norm(x - x_old, ord=float("+inf"), axis=0).max().item()
        progress.set_postfix(max_diff=max_diff)
        if max_diff < epsilon:
            logger.debug(f"Converged after {i} iterations up to {epsilon}.")
            break
        x_old = x
    else:  # for/else, cf. https://book.pythontips.com/en/latest/for_-_else.html
        logger.warning(f"No convergence after {max_iter} iterations with epsilon={epsilon}.")
    return x


def validate_x0(x0: torch.Tensor) -> None:
    """
    Validate the initial value $x$.

    In particular, the method checks that

    - all entries are between 0 and 1
    - the entries sum to 1 (along the first dimension)

    :param x0:
        the initial value.

    :raises ValueError:
        if the input is invalid.
    """
    if (x0 < 0.0).any() or (x0 > 1.0).any():
        raise ValueError("Encountered values outside of [0, 1].")
    x_sum = x0.sum(dim=0)
    if not torch.allclose(x_sum, torch.ones_like(x_sum)):
        raise ValueError("The entries do not sum to 1.")


def prepare_x0(
    x0: Optional[torch.Tensor] = None, indices: Collection[int] = None, n: Optional[int] = None
) -> torch.Tensor:
    """
    Prepare a start value.

    :param x0:
        the start value. It will be passed through. First preference.
    :param indices:
        a number of indices. This will create a (n, len(indices)) one-hot vector. Second preference.
    :param n:
        the number of nodes. If neither of the other options is given, it will create a constant $1/n$ vector.

    :param x0: shape: `(n,)` or `(n, batch_size)`
        the initial value $x$
    """
    if x0 is not None:
        return x0
    if indices is not None:
        x0 = torch.zeros(n, len(indices))
        x0[indices] = 1.0
        return x0
    return torch.full(size=(n,), fill_value=1.0 / n)


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

    :return: shape: `(n,)` or `(n, batch_size)`
        the page-rank vector, i.e., a score between 0 and 1 for each node.

    :raises ValueError:
        if neither `adj` nor `edge_index` are provided
    """
    # normalize inputs
    adj = prepare_page_rank_adjacency(adj=adj, edge_index=edge_index)
    x0 = prepare_x0(x0=x0, n=adj.shape[0])

    # input normalization
    validate_x0(x0=x0)

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
