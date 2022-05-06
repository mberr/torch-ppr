"""Utility functions."""
import logging
from typing import Collection, Optional, Union

import torch
from torch.nn import functional
from tqdm.auto import tqdm

__all__ = [
    "DeviceHint",
    "resolve_device",
    "prepare_num_nodes",
    "edge_index_to_sparse_matrix",
    "prepare_page_rank_adjacency",
    "validate_x",
    "prepare_x0",
    "power_iteration",
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


def prepare_num_nodes(edge_index: torch.Tensor, num_nodes: Optional[int] = None) -> int:
    """
    Prepare the number of nodes.

    :param edge_index: shape: $(2, m)$
        the edge index
    :param num_nodes:
        the number of nodes. If None, it is inferred from `edge_index`.

    :return:
        the number of nodes
    """
    if num_nodes is not None:
        return num_nodes

    num_nodes = edge_index.max().item() + 1
    logger.info(f"Inferred num_nodes={num_nodes}")
    return num_nodes


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
    num_nodes = prepare_num_nodes(edge_index=edge_index, num_nodes=num_nodes)
    return torch.sparse_coo_tensor(
        indices=edge_index,
        values=torch.ones_like(edge_index[0], dtype=torch.get_default_dtype()),
        size=(num_nodes, num_nodes),
    )


def validate_adjacency(adj: torch.Tensor, n: Optional[int] = None):
    """
    Validate the page-rank adjacency matrix.

    In particular, the method checks that

    - the shape is (n, n)
    - the row-sum is 1

    :param adj: shape: (n, n)
        the adjacency matrix
    :param n:
        the number of nodes

    :raises ValueError:
        if the adjacency matrix is invalid
    """
    # check shape
    if n is None:
        n = adj.shape[0]
    if adj.shape != (n, n):
        raise ValueError(f"Invalid shape: {adj.shape}. expected: {(n, n)}")

    # check row-sum
    adj_sum = torch.sparse.sum(adj, dim=1).to_dense()
    if not torch.allclose(adj_sum, torch.ones_like(adj_sum)):
        raise ValueError(f"Invalid row sum: {adj_sum}. expected 1.0")


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
    # adjacency normalization: normalize to row-sum = 1
    degree_inv = torch.reciprocal(torch.sparse.sum(adj, dim=0).to_dense())
    degree_inv = torch.sparse_coo_tensor(
        indices=torch.arange(degree_inv.shape[0], device=adj.device).unsqueeze(dim=0).repeat(2, 1),
        values=degree_inv,
    )
    return torch.sparse.mm(mat1=degree_inv, mat2=adj)


def validate_x(x: torch.Tensor, n: Optional[int] = None) -> None:
    """
    Validate a (batched) page-rank vector.

    In particular, the method checks that

    - the tensor dimension is (n,) or (n, batch_size)
    - all entries are between 0 and 1
    - the entries sum to 1 (along the first dimension)

    :param x:
        the initial value.
    :param n:
        the number of nodes.

    :raises ValueError:
        if the input is invalid.
    """
    if x.ndim > 2 or (n is not None and x.shape[0] != n):
        raise ValueError(f"Invalid shape: {x.shape}")

    if (x < 0.0).any() or (x > 1.0).any():
        raise ValueError(
            f"Encountered values outside of [0, 1]. min={x.min().item()}, max={x.max().item()}"
        )

    x_sum = x.sum(dim=0)
    if not torch.allclose(x_sum, torch.ones_like(x_sum)):
        raise ValueError(f"The entries do not sum to 1. {x_sum[x_sum != 0]}")


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

    :raises ValueError:
        if neither `x0` nor `n` are provided

    :return: shape: `(n,)` or `(n, batch_size)`
        the initial value $x$
    """
    if x0 is not None:
        return x0
    if n is None:
        raise ValueError("If x0 is not provided, n must be given.")
    if indices is not None:
        k = len(indices)
        x0 = torch.zeros(n, k)
        x0[indices, torch.arange(k, device=x0.device)] = 1.0
        return x0
    return torch.full(size=(n,), fill_value=1.0 / n)


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
    :param max_iter: $0 < max_iter$
        the maximum number of iterations
    :param epsilon: $epsilon > 0$
        a (small) constant to check for convergence
    :param use_tqdm:
        whether to use a tqdm progress bar
    :param device:
        the device to use, or a hint thereof

    :return: shape: $(n,)$ or $(n, batch_size)$
        the $x$ value after convergence (or maximum number of iterations).
    """
    # normalize device
    device = resolve_device(device=device)
    # send tensors to device
    adj = adj.to(device=device)
    x0 = x0.to(device=device)
    no_batch = x0.ndim < 2
    if no_batch:
        x0 = x0.unsqueeze(dim=-1)
    # power iteration
    x_old = x = x0
    beta = 1.0 - alpha
    progress = tqdm(range(max_iter), unit_scale=True, leave=False, disable=not use_tqdm)
    for i in progress:
        # calculate x = (1 - alpha) * A.dot(x) + alpha * x0
        x = torch.sparse.addmm(input=x0, sparse=adj, dense=x, beta=alpha, alpha=beta)
        # note: while the adjacency matrix should already be row-sum normalized,
        #       we additionally normalize x to avoid accumulating errors due to loss of precision
        x = functional.normalize(x, dim=0, p=1)
        # calculate difference, shape: (batch_size,)
        diff = torch.linalg.norm(x - x_old, ord=float("+inf"), axis=0)
        mask = diff > epsilon
        if use_tqdm:
            progress.set_postfix(
                max_diff=diff.max().item(), converged=1.0 - mask.float().mean().item()
            )
        if not mask.any():
            logger.debug(f"Converged after {i} iterations up to {epsilon}.")
            break
        x_old = x
    else:  # for/else, cf. https://book.pythontips.com/en/latest/for_-_else.html
        logger.warning(f"No convergence after {max_iter} iterations with epsilon={epsilon}.")
    if no_batch:
        x = x.squeeze(dim=-1)
    return x