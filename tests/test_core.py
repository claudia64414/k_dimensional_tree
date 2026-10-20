"""Tests for k_dimensional_tree.core.KDTree."""

import math
import unittest

from k_dimensional_tree import KDTree, Point


class TestConstruction(unittest.TestCase):
    def test_empty_tree(self):
        t = KDTree()
        self.assertIsNone(t.k)
        self.assertEqual(len(t), 0)
        self.assertIsNone(t.nearest_neighbor((0.0, 0.0)))
        self.assertEqual(t.range_query((0.0, 0.0), (1.0, 1.0)), [])

    def test_initial_points(self):
        pts = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]
        t = KDTree(pts)
        self.assertEqual(t.k, 2)
        self.assertEqual(len(t), len(pts))

    def test_insert_after_empty(self):
        t = KDTree()
        t.insert((1, 2, 3))
        self.assertEqual(t.k, 3)
        self.assertEqual(len(t), 1)

    def test_dimension_mismatch_raises(self):
        t = KDTree([(1, 2)])
        with self.assertRaises(ValueError):
            t.insert((1, 2, 3))

    def test_non_tuple_raises(self):
        t = KDTree()
        with self.assertRaises(TypeError):
            t.insert([1, 2])  # type: ignore[arg-type]


class TestNearestNeighbor(unittest.TestCase):
    def test_single_point(self):
        t = KDTree([(5, 5)])
        self.assertEqual(t.nearest_neighbor((0, 0)), (5, 5))

    def test_exact_match(self):
        pts = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]
        t = KDTree(pts)
        for p in pts:
            self.assertEqual(t.nearest_neighbor(p), p)

    def test_2d_classic(self):
        # Well-known example dataset.
        pts = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]
        t = KDTree(pts)
        self.assertEqual(t.nearest_neighbor((9, 2)), (8, 1))
        self.assertEqual(t.nearest_neighbor((6, 1)), (7, 2))
        self.assertEqual(t.nearest_neighbor((3, 6)), (4, 7))

    def test_duplicate_points(self):
        pts = [(1, 1), (1, 1), (1, 1)]
        t = KDTree(pts)
        self.assertEqual(t.nearest_neighbor((1, 1)), (1, 1))
        self.assertEqual(len(t), 3)

    def test_target_dimension_mismatch(self):
        t = KDTree([(1, 2)])
        with self.assertRaises(ValueError):
            t.nearest_neighbor((1, 2, 3))

    def test_brute_force_comparison(self):
        # Deterministic pseudo-random points via a simple LCG.
        seed = 42
        state = seed
        pts = []
        for _ in range(50):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            x = state % 1000
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            y = state % 1000
            pts.append((x, y))
        t = KDTree(pts)
        for q in [(500, 500), (0, 0), (999, 999), (123, 456), (777, 222)]:
            expected = min(pts, key=lambda p: (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)
            got = t.nearest_neighbor(q)
            self.assertEqual(got, expected, f"failed for query {q}")

    def test_returns_genuine_point_object(self):
        pts = [(1, 2), (3, 4)]
        t = KDTree(pts)
        result = t.nearest_neighbor((0, 0))
        self.assertIn(result, pts)


class TestRangeQuery(unittest.TestCase):
    def test_empty_tree_returns_empty(self):
        t = KDTree()
        self.assertEqual(t.range_query((0, 0), (10, 10)), [])

    def test_all_points_in_range(self):
        pts = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]
        t = KDTree(pts)
        result = t.range_query((0, 0), (100, 100))
        self.assertEqual(len(result), len(pts))
        for p in pts:
            self.assertIn(p, result)

    def test_tight_box_selects_one(self):
        pts = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]
        t = KDTree(pts)
        result = t.range_query((9, 0), (9, 7))
        self.assertEqual(result, [(9, 6)])

    def test_box_selects_multiple(self):
        pts = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]
        t = KDTree(pts)
        result = set(t.range_query((4, 2), (7, 7)))
        self.assertEqual(result, {(5, 4), (4, 7), (7, 2)})

    def test_degenerate_box_single_point(self):
        pts = [(2, 3), (5, 4), (9, 6)]
        t = KDTree(pts)
        self.assertEqual(t.range_query((5, 4), (5, 4)), [(5, 4)])

    def test_box_outside_all_points(self):
        pts = [(1, 1), (2, 2)]
        t = KDTree(pts)
        self.assertEqual(t.range_query((10, 10), (20, 20)), [])

    def test_range_dimension_mismatch(self):
        t = KDTree([(1, 2)])
        with self.assertRaises(ValueError):
            t.range_query((0,), (1, 2))
        with self.assertRaises(ValueError):
            t.range_query((0, 0), (1,))


class TestPointAlias(unittest.TestCase):
    def test_point_is_exported(self):
        self.assertIs(Point, tuple)


if __name__ == "__main__":
    unittest.main()
