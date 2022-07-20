"""Tests for the API."""
import unittest
from typing import Counter, Optional, Tuple

import pytest
import torch
from torch.nn import functional

from torch_ppr import utils


def test_resolve_device():
    """Test for resolving devices."""
    for hint, device in (
        (None, torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")),
        ("cpu", torch.device("cpu")),
        (torch.device("cpu"), torch.device("cpu")),
    ):
        assert device == utils.resolve_device(device=hint)


class UtilsTest(unittest.TestCase):
    """Test utilities."""

    num_nodes: int = 7
    num_edges: int = 33

    def setUp(self) -> None:
        """Prepare data."""
        # fix seed for reproducible tests
        torch.manual_seed(seed=42)
        self.edge_index = torch.cat(
            [
                torch.randint(self.num_nodes, size=(2, self.num_edges - self.num_nodes)),
                # ensure connectivity
                torch.arange(self.num_nodes).unsqueeze(0).repeat(2, 1),
            ],
            dim=-1,
        )
        target_indices = self.edge_index[1].tolist()
        counts = Counter(target_indices)
        values = torch.as_tensor([1.0 / counts[i] for i in target_indices])
        self.adj = torch.sparse_coo_tensor(indices=self.edge_index, values=values)

    def _verify_adjacency(self, adj: torch.Tensor):
        assert torch.is_tensor(adj)
        assert adj.shape == (self.num_nodes, self.num_nodes)

    def test_prepare_num_nodes(self):
        """Test inferring the number of nodes from an edge index."""
        for num_nodes in (None, self.num_nodes):
            assert (
                utils.prepare_num_nodes(edge_index=self.edge_index, num_nodes=num_nodes)
                == self.num_nodes
            )

    def test_edge_index_to_sparse_matrix(self):
        """Test conversion of edge indices to sparse matrices."""
        for num_nodes_ in (self.num_nodes, None):
            adj = utils.edge_index_to_sparse_matrix(
                edge_index=self.edge_index,
                num_nodes=num_nodes_,
            )
            assert adj.shape == (self.num_nodes, self.num_nodes)

    def test_validate_adjacancy(self):
        """Test adjacency validation."""
        adj = utils.prepare_page_rank_adjacency(edge_index=self.edge_index)
        # plain validation with shape inference
        utils.validate_adjacency(adj=adj)
        # plain validation with explicit shape
        utils.validate_adjacency(adj=adj, n=self.num_nodes)
        # validation with CSR matrix
        utils.validate_adjacency(adj=adj.to_sparse_csr())
        # test error raising
        for adj in (
            # an edge_index instead of adj
            self.edge_index,
            # wrong shape
            torch.sparse_coo_tensor(
                indices=torch.empty(2, 0, dtype=torch.long),
                values=torch.empty(0),
                size=(2, 3),
            ),
            # wrong value range
            torch.sparse_coo_tensor(
                indices=self.edge_index,
                values=torch.full(size=(self.num_edges,), fill_value=2.0),
                size=(self.num_nodes, self.num_nodes),
            ),
            # wrong sum
            torch.sparse_coo_tensor(
                indices=self.edge_index,
                values=torch.ones(self.num_edges),
                size=(self.num_nodes, self.num_nodes),
            ),
        ):
            with self.assertRaises(ValueError):
                utils.validate_adjacency(adj=adj)

    def test_prepare_page_rank_adjacency(self):
        """Test adjacency preparation."""
        for (adj, edge_index, add_identity) in (
            # from edge index
            (None, self.edge_index, False),
            # passing through adjacency matrix
            (self.adj, None, False),
            (self.adj, self.edge_index, False),
            # add identity
            (None, self.edge_index, True),
        ):
            adj2 = utils.prepare_page_rank_adjacency(
                adj=adj, edge_index=edge_index, add_identity=add_identity
            )
            utils.validate_adjacency(adj=adj2, n=self.num_nodes)
            if adj is not None:
                assert adj is adj2

    def _valid_x0(self, size: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        """Generate a valid x0."""
        size = size or (self.num_nodes,)
        return functional.normalize(torch.rand(size=size), p=1, dim=0)

    def test_validate_x(self):
        """Test page-rank vector validation."""
        # valid single
        x0_valid = self._valid_x0()
        utils.validate_x(x=x0_valid, n=self.num_nodes)
        # valid batch
        x0_valid_batch = self._valid_x0(size=(self.num_nodes, 12))
        utils.validate_x(x=x0_valid_batch, n=self.num_nodes)
        # invalid shape, wrong dim
        with self.assertRaises(ValueError):
            utils.validate_x(x=x0_valid, n=self.num_nodes + 1)
        # invalid shape, too many dim
        with self.assertRaises(ValueError):
            utils.validate_x(x=x0_valid_batch[..., None], n=self.num_nodes)
        # too large value
        for value in (-1.0, 2.0):
            with self.assertRaises(ValueError):
                x0_invalid = x0_valid.clone()
                x0_invalid[0] = value
                utils.validate_x(x=x0_invalid, n=self.num_nodes)

    def test_prepare_x0(self):
        """Test x0 preparation."""
        for x0, indices in (
            # x0 pass-through
            (self._valid_x0(), None),
            (self._valid_x0(size=(self.num_nodes, 12)), None),
            (self._valid_x0(), [1, 2]),
            # indices
            (None, [1, 2]),
            # only n
            (None, None),
        ):
            x0 = utils.prepare_x0(x0, indices, n=self.num_nodes)
            utils.validate_x(x0, n=self.num_nodes)

    def test_power_iteration(self):
        """Test power-iteration."""
        adj = utils.prepare_page_rank_adjacency(edge_index=self.edge_index)
        for x0 in (self._valid_x0(), self._valid_x0(size=(self.num_nodes, 12))):
            x = utils.power_iteration(adj=adj, x0=x0, max_iter=5)
            utils.validate_x(x=x, n=self.num_nodes)

    def test_batched_personalized_page_rank(self):
        """Test batched PPR calculation."""
        x = utils.batched_personalized_page_rank(
            adj=self.adj, indices=torch.arange(self.num_nodes), batch_size=self.num_nodes // 3
        )
        utils.validate_x(x)


@pytest.mark.parametrize("n", [8, 16])
def test_sparse_diagonal(n: int):
    """Test for sparse diagonal matrix creation."""
    values = torch.rand(n)
    matrix = utils.sparse_diagonal(values=values)
    assert torch.is_tensor(matrix)
    assert matrix.shape == (n, n)
    assert matrix.is_sparse
    assert torch.allclose(matrix.to_dense(), torch.diag(values))
