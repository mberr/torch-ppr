# -*- coding: utf-8 -*-

"""Trivial version test."""

import unittest

from torch_ppr.version import get_version


class TestVersion(unittest.TestCase):
    """Trivially test a version."""

    def test_version_type(self):
        """Test the version is a string.

        This is only meant to be an example test.
        """
        version = get_version(with_git_hash=True)
        self.assertIsInstance(version, str)
