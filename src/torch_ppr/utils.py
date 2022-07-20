"""Utility functions."""
import logging
from typing import Any, Collection, Mapping, Optional, Union

import torch
from torch.nn import functional
from torch_max_mem import MemoryUtilizationMaximizer
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
    "batched_personalized_page_rank",
]

logger = logging.getLogger(__name__)

DeviceHint = Union[None, str, torch.device]


def resolve_device(device: DeviceHint = None) -> torch.device:
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

    If an explicit number is given, this number will be used. Otherwise, infers the number of nodes as the maximum id
    in the edge index.

    :param edge_index: shape: ``(2, m)``
        the edge index
    :param num_nodes:
        the number of nodes. If ``None``, it is inferred from ``edge_index``.

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

    Uses the edge index for non-zero entries, and fills in ``1`` as entries.

    :param edge_index: shape: ``(2, m)``
        the edge index
    :param num_nodes:
        the number of nodes used to determine the shape of the matrix.
        If ``None``, it is inferred from ``edge_index``.

    :return: shape: ``(n, n)``
        the adjacency matrix as a sparse tensor, cf. :func:`torch.sparse_coo_tensor`.
    """
    num_nodes = prepare_num_nodes(edge_index=edge_index, num_nodes=num_nodes)
    return torch.sparse_coo_tensor(
        indices=edge_index,
        values=torch.ones_like(edge_index[0], dtype=torch.get_default_dtype()),
        size=(num_nodes, num_nodes),
    )


def validate_adjacency(adj: torch.Tensor, n: Optional[int] = None, rtol: float = 1.0e-04):
    """
    Validate the page-rank adjacency matrix.

    In particular, the method checks that

    - the shape is ``(n, n)``
    - the row-sum is ``1``

    :param adj: shape: ``(n, n)``
        the adjacency matrix
    :param n:
        the number of nodes
    :param rtol:
        the tolerance for checking the sum is close to 1.0

    :raises ValueError:
        if the adjacency matrix is invalid
    """
    # check dtype
    if not torch.is_floating_point(adj):
        if adj.shape[0] == 2 and adj.shape[1] != 2:
            logger.warning(
                "The passed adjacency matrix looks like an edge_index; did you pass it for the wrong parameter?"
            )
        raise ValueError(
            f"Invalid adjacency matrix data type: {adj.dtype}, should be a floating dtype."
        )

    # check shape
    if n is None:
        n = adj.shape[0]
    if adj.shape != (n, n):
        raise ValueError(f"Invalid adjacency matrix shape: {adj.shape}. expected: {(n, n)}")

    # check value range
    if adj.is_sparse and not adj.is_sparse_csr:
        adj = adj.coalesce()
    values = adj.values()
    if (values < 0.0).any() or (values > 1.0).any():
        raise ValueError(
            f"Invalid values outside of [0, 1]: min={values.min().item()}, max={values.max().item()}"
        )

    # check column-sum
    if adj.is_sparse and not adj.is_sparse_csr:
        adj_sum = torch.sparse.sum(adj, dim=0).to_dense()
    else:
        # hotfix until torch.sparse.sum is implemented
        adj_sum = adj.t() @ torch.ones(adj.shape[0])
    exp_sum = torch.ones_like(adj_sum)
    mask = adj_sum == 0
    if mask.any():
        logger.warning(f"Adjacency contains {mask.sum().item()} isolated nodes.")
        exp_sum[mask] = 0.0
    if not torch.allclose(adj_sum, exp_sum, rtol=rtol):
        raise ValueError(
            f"Invalid column sum: {adj_sum} (min: {adj_sum.min().item()}, max: {adj_sum.max().item()}). "
            f"Expected 1.0 with a relative tolerance of {rtol}.",
        )


def sparse_diagonal(values: torch.Tensor) -> torch.Tensor:
    """Create a sparse diagonal matrix with the given values.

    :param values: shape: ``(n,)``
        the values

    :return: shape: ``(n, n)``
        a sparse diagonal matrix
    """
    return torch.sparse_coo_tensor(
        indices=torch.arange(values.shape[0], device=values.device).unsqueeze(dim=0).repeat(2, 1),
        values=values,
    )


def prepare_page_rank_adjacency(
    adj: Optional[torch.Tensor] = None,
    edge_index: Optional[torch.LongTensor] = None,
    num_nodes: Optional[int] = None,
    add_identity: bool = False,
) -> torch.Tensor:
    """
    Prepare the page-rank adjacency matrix.

    If no explicit adjacency is given, the methods first creates an adjacency matrix from the edge index,
    cf. :func:`edge_index_to_sparse_matrix`. Next, the matrix is symmetrized as

    .. math::
        A := A + A^T

    Finally, the matrix is normalized such that the columns sum to one.

    :param adj: shape: ``(n, n)``
        the adjacency matrix
    :param edge_index: shape: ``(2, m)``
        the edge index
    :param num_nodes:
        the number of nodes used to determine the shape of the adjacency matrix.
        If ``None``, and ``adj`` is not already provided, it is inferred from ``edge_index``.
    :param add_identity:
        whether to add an identity matrix to ``A`` to ensure that each node has a degree of at least one.

    :raises ValueError:
        if neither is provided, or the adjacency matrix is invalid

    :return: shape: ``(n, n)``
        the symmetric, normalized, and sparse adjacency matrix
    """
    if adj is not None:
        return adj

    if edge_index is None:
        raise ValueError("Must provide at least one of `adj` and `edge_index`.")

    # convert to sparse matrix, shape: (n, n)
    adj = edge_index_to_sparse_matrix(edge_index=edge_index, num_nodes=num_nodes)
    # symmetrize
    adj = adj + adj.t()
    # add identity matrix if requested
    if add_identity:
        adj = adj + sparse_diagonal(torch.ones(adj.shape[0], dtype=adj.dtype, device=adj.device))

    # adjacency normalization: normalize to col-sum = 1
    degree_inv = torch.reciprocal(
        torch.sparse.sum(adj, dim=0).to_dense().clamp_min(min=torch.finfo(adj.dtype).eps)
    )
    degree_inv = sparse_diagonal(values=degree_inv)
    return torch.sparse.mm(adj, degree_inv)


def validate_x(x: torch.Tensor, n: Optional[int] = None) -> None:
    """
    Validate a (batched) page-rank vector.

    In particular, the method checks that

    - the tensor dimension is ``(n,)`` or ``(n, batch_size)``
    - all entries are between ``0`` and ``1``
    - the entries sum to ``1`` (along the first dimension)

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

    The following precedence order is used:

    1. an explicit start value, via ``x0``. If present, this tensor is passed through without further modification.
    2. a one-hot matrix created via ``indices``. The matrix is of shape ``(n, len(indices))`` and has a single 1 per
       column at the given indices.
    3. a uniform ``1/n`` vector of shape ``(n,)``

    :param x0:
        the start value.
    :param indices:
        a non-zero indices
    :param n:
        the number of nodes

    :raises ValueError:
        if neither ``x0`` nor ``n`` are provided

    :return: shape: ``(n,)`` or ``(n, batch_size)``
        the initial value ``x``
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

    :param adj: shape: ``(n, n)``
        the (sparse) adjacency matrix
    :param x0: shape: ``(n,)``, or ``(n, batch_size)``
        the initial value for ``x``.
    :param alpha: ``0 < alpha < 1``
        the smoothing value / teleport probability
    :param max_iter: ``0 < max_iter``
        the maximum number of iterations
    :param epsilon: ``epsilon > 0``
        a (small) constant to check for convergence
    :param use_tqdm:
        whether to use a tqdm progress bar
    :param device:
        the device to use, or a hint thereof, cf. :func:`resolve_device`

    :return: shape: ``(n,)`` or ``(n, batch_size)``
        the ``x`` value after convergence (or maximum number of iterations).
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
        x = torch.sparse.addmm(
            # dense matrix to be added
            x0,
            # sparse matrix to be multiplied
            adj,
            # dense matrix to be multiplied
            x,
            # multiplier for added matrix
            beta=alpha,
            # multiplier for product
            alpha=beta,
        )
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


def _ppr_hasher(kwargs: Mapping[str, Any]) -> int:
    # assumption: batched PPR memory consumption only depends on the matrix A,
    # in particular, the shape and the number of nonzero elements
    adj: torch.Tensor = kwargs.get("adj")
    return hash((adj.shape[0], getattr(adj, "nnz", adj.numel())))


ppr_maximizer = MemoryUtilizationMaximizer(hasher=_ppr_hasher)


@ppr_maximizer
def batched_personalized_page_rank(
    adj: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    **kwargs,
) -> torch.Tensor:
    """
    Batch-wise PPR computation with automatic memory optimization.

    :param adj: shape: ``(n, n)``
        the adjacency matrix.
    :param indices: shape: ``k``
        the indices for which to compute PPR
    :param batch_size: ``batch_size > 0``
        the batch size. Will be reduced if necessary
    :param kwargs:
        additional keyword-based parameters passed to :func:`power_iteration`

    :return: shape: ``(n, k)``
        the PPR vectors for each node index
    """
    return torch.cat(
        [
            power_iteration(adj=adj, x0=prepare_x0(indices=indices_batch, n=adj.shape[0]), **kwargs)
            for indices_batch in torch.split(indices, batch_size)
        ],
        dim=1,
    )
