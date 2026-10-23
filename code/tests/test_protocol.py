import numpy as np

from clusterization.data import clean_records
from clusterization.sampling import annotation_order, reserve_test_split
from clusterization.selection import choose


def test_group_split_keeps_all_linked_documents_together():
    groups = np.repeat(np.arange(12), [1, 2, 3, 1, 5, 2, 2, 3, 4, 1, 2, 4])
    reserve, test = reserve_test_split(groups, 31415)
    assert set(groups[reserve]).isdisjoint(groups[test])
    assert sorted([*reserve, *test]) == list(range(len(groups)))


def test_missing_class_sampling_has_nested_budgets_without_replacement():
    y = np.repeat(np.arange(8), 100)
    order, info = annotation_order(y, "missing_classes", 200, np.random.default_rng(9))
    assert len(np.unique(order)) == len(order)
    assert len(np.unique(y[order])) == 4
    assert set(order[:20]) <= set(order[:50]) <= set(order[:200])
    assert set(y[order]) == set(info["favored_classes"])


def test_adaptive_rule_uses_paired_difference_not_marginal_uncertainty():
    # Both candidates vary strongly, but their difference is constant.
    partial = np.array([0.5, 0.4])
    bootstrap = np.array([[0.3, 0.2], [0.5, 0.4], [0.7, 0.6]])
    result = choose("M6", partial, np.array([0.1, 0.9]), np.array([0.1, 0.9]), bootstrap)
    assert result["selected"] == 0
    assert result["eligible_count"] == 1


def test_exact_duplicates_union_known_groups_and_remove_conflicts():
    rows = [dict(id="a", group="g1", text="same  text", label="x"),
            dict(id="b", group="g2", text="same text", label="x"),
            dict(id="c", group="g2", text="related", label="x"),
            dict(id="d", group="g3", text="conflict", label="x"),
            dict(id="e", group="g4", text="conflict", label="y"),
            dict(id="f", group="g5", text="other one", label="y"),
            dict(id="g", group="g6", text="other two", label="y")]
    clean, stats = clean_records(rows, 20, 0)
    assert stats["conflicting_duplicate_rows"] == 2
    assert stats["duplicates_removed"] == 1
    lookup = {r["id"]: r for r in clean}
    assert lookup["a"]["group"] == lookup["c"]["group"]
