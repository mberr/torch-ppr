# -*- coding: utf-8 -*-

"""(Personalized) Page-Rank computation using PyTorch."""

from .api import page_rank, personalized_page_rank

__all__ = [
    "page_rank",
    "personalized_page_rank",
]
