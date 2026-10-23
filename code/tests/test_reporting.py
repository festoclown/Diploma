"""Reporting identifiers accept NumPy scalars returned by pandas groupby."""

import numpy as np

from clusterization.storage import derived_seed, digest


def test_report_group_values_hash_like_python_scalars():
    group = ["series", "synthetic", np.int64(20), np.float64(0.5)]
    plain = ["series", "synthetic", 20, 0.5]
    assert digest(group) == digest(plain)
    assert derived_seed(group, "M1", "paired_ci") == derived_seed(
        plain, "M1", "paired_ci"
    )
