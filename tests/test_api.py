"""Tests for public API."""

import unittest

import torch

from torch_ppr import page_rank, personalized_page_rank, utils


class APITest(unittest.TestCase):
    """Test public API."""

    num_nodes: int = 7
    num_edges: int = 33

    def setUp(self) -> None:
        """Prepare data."""
        generator = torch.manual_seed(42)
        self.edge_index = torch.cat(
            [
                torch.randint(
                    self.num_nodes, size=(2, self.num_edges - self.num_nodes), generator=generator
                ),
                # ensure connectivity
                torch.arange(self.num_nodes).unsqueeze(0).repeat(2, 1),
            ],
            dim=-1,
        )
        self.adj = utils.prepare_page_rank_adjacency(edge_index=self.edge_index)

    def test_page_rank_edge_index(self):
        """Test Page Rank calculation for an adjacency given as edge list."""
        page_rank(edge_index=self.edge_index)

    def test_page_rank_adj(self):
        """Test Page Rank calculation."""
        page_rank(adj=self.adj)

    def test_personalized_page_rank_edge_index(self):
        """Test Personalized Page Rank calculation for an adjacency given as edge list."""
        personalized_page_rank(edge_index=self.edge_index)

    def test_personalized_page_rank_adj(self):
        """Test Personalized Page Rank calculation."""
        personalized_page_rank(adj=self.adj)

    def test_page_rank_manual(self):
        """Test Page Rank calculation on a simple manually created example."""
        # A - B - C
        #     |
        #     D
        edge_index = torch.as_tensor(data=[(0, 1), (1, 2), (1, 3)]).t()
        x = page_rank(edge_index=edge_index)
        # verify that central node has the largest PR value
        assert x.argmax() == 1

    def test_page_rank_isolated_vertices(self):
        """Test Page-Rank with isolated vertices."""
        # create isolated node, ID=0
        edge_index = self.edge_index + 1
        x = page_rank(edge_index=edge_index, add_identity=True)
        # isolated node has only one self-loop -> no change in mass to initial mass
        self.assertAlmostEqual(x[0].item(), 1 / (self.num_nodes + 1))
        # verify that other nodes are unaffected
        x2 = page_rank(edge_index=self.edge_index)
        # rescale
        x2 = x2 * (self.num_nodes / (self.num_nodes + 1))
        assert torch.allclose(x2, x[1:], atol=1.0e-02)
