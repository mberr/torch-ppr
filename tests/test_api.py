"""Tests for public API."""

import unittest

import torch

from torch_ppr import api


class APITest(unittest.TestCase):
    """Test public API."""

    num_nodes: int = 7
    num_edges: int = 33

    def setUp(self) -> None:
        """Prepare data."""
        self.edge_index = torch.cat(
            [
                torch.randint(self.num_nodes, size=(2, self.num_edges - self.num_nodes)),
                # ensure connectivity
                torch.arange(self.num_nodes).unsqueeze(0).repeat(2, 1),
            ],
            dim=-1,
        )

    def test_page_rank(self):
        """Test Page Rank calculation."""
        api.page_rank(edge_index=self.edge_index)

    def test_personalized_page_rank(self):
        """Test Personalized Page Rank calculation."""
        api.personalized_page_rank(edge_index=self.edge_index)

    def test_page_rank_manual(self):
        """Test Page Rank calculation on a simple manually created example."""
        # A - B - C
        #     |
        #     D
        edge_index = torch.as_tensor(data=[(0, 1), (1, 2), (1, 3)]).t()
        x = api.page_rank(edge_index=edge_index)
        # verify that central node has the largest PR value
        assert x.argmax() == 1
