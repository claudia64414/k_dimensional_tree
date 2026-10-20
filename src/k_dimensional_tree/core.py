"""Core implementation of a k-dimensional tree.

A k-d tree partitions space along axis-aligned planes, alternating the
splitting axis at each level. This gives O(log n) average-case nearest
neighbor and orthogonal range search when points are well-distributed.

Design decisions
----------------
* Points are tuples of int/float. We store them verbatim so the caller's
  objects come back unchanged.
* Duplicates are allowed. We break ties on equal coordinates by always
  recursing into the left subtree before the right, which keeps behavior
  deterministic regardless of insertion order for the *set* of returned
  points (the traversal order is insertion order for duplicates).
* Distance comparisons use squared distance to avoid sqrt, which is
  both faster and avoids float-precision differences in tests.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

Point = tuple


class _Node:
    __slots__ = ("point", "left", "right", "axis")

    def __init__(self, point: Point, axis: int):
        self.point = point
        self.axis = axis
        self.left: Optional[_Node] = None
        self.right: Optional[_Node] = None


class KDTree:
    """A k-dimensional tree supporting nearest-neighbor and range queries.

    Parameters
    ----------
    points : iterable of tuples, optional
        Initial points to insert. All points must have the same length.
        Inserted one at a time, so the resulting tree shape depends on order.
    """

    def __init__(self, points: Optional[Sequence[Point]] = None):
        self._root: Optional[_Node] = None
        self._size = 0
        self._k: Optional[int] = None
        if points:
            for p in points:
                self.insert(p)

    @property
    def k(self) -> Optional[int]:
        """Dimensionality of points in this tree, or None if empty."""
        return self._k

    def __len__(self) -> int:
        return self._size

    def insert(self, point: Point) -> None:
        """Insert a single point into the tree.

        Raises ValueError if the point's dimensionality does not match
        existing points.
        """
        if not isinstance(point, tuple):
            raise TypeError(f"point must be a tuple, got {type(point).__name__}")
        if self._k is None:
            self._k = len(point)
        elif len(point) != self._k:
            raise ValueError(
                f"point has dimension {len(point)}, expected {self._k}"
            )
        self._root = self._insert(self._root, point, 0)
        self._size += 1

    def _insert(
        self, node: Optional[_Node], point: Point, depth: int
    ) -> _Node:
        if node is None:
            axis = depth % self._k  # type: ignore[operator]
            return _Node(point, axis)
        if point[node.axis] < node.point[node.axis]:
            node.left = self._insert(node.left, point, depth + 1)
        else:
            # Equal coordinate goes right: deterministic and keeps the tree
            # connected without special-casing duplicates.
            node.right = self._insert(node.right, point, depth + 1)
        return node

    def nearest_neighbor(self, target: Point) -> Optional[Point]:
        """Return the closest point to `target`, or None if the tree is empty."""
        if self._root is None or self._k is None:
            return None
        if len(target) != self._k:
            raise ValueError(
                f"target has dimension {len(target)}, expected {self._k}"
            )
        best: List[Optional[Point]] = [None]
        best_sq: List[float] = [float("inf")]
        self._nn(self._root, target, best, best_sq)
        return best[0]

    def _nn(
        self,
        node: Optional[_Node],
        target: Point,
        best: List[Optional[Point]],
        best_sq: List[float],
    ) -> None:
        if node is None:
            return
        d = self._sq_dist(node.point, target)
        if d < best_sq[0] or (d == best_sq[0] and best[0] is None):
            best_sq[0] = d
            best[0] = node.point
        axis = node.axis
        diff = target[axis] - node.point[axis]
        # Visit the closer side first so we can prune the farther side.
        near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)
        self._nn(near, target, best, best_sq)
        # Only explore the far side if it could contain a closer point.
        if diff * diff < best_sq[0]:
            self._nn(far, target, best, best_sq)

    def range_query(self, low: Point, high: Point) -> List[Point]:
        """Return all points p with low[i] <= p[i] <= high[i] for every i.

        Points are returned in the order they are encountered by an
        in-order traversal of the tree (i.e., roughly left-to-right in the
        first splitting axis).
        """
        if self._root is None or self._k is None:
            return []
        if len(low) != self._k or len(high) != self._k:
            raise ValueError("range bounds have wrong dimensionality")
        result: List[Point] = []
        self._range(self._root, low, high, result)
        return result

    def _range(
        self,
        node: Optional[_Node],
        low: Point,
        high: Point,
        result: List[Point],
    ) -> None:
        if node is None:
            return
        axis = node.axis
        p = node.point
        # If the node's coordinate on this axis is below the lower bound,
        # nothing in the left subtree can be in range.
        if p[axis] >= low[axis]:
            self._range(node.left, low, high, result)
        if all(low[i] <= p[i] <= high[i] for i in range(self._k)):  # type: ignore[arg-type]
            result.append(p)
        if p[axis] <= high[axis]:
            self._range(node.right, low, high, result)

    @staticmethod
    def _sq_dist(a: Point, b: Point) -> float:
        s = 0.0
        for i in range(len(a)):
            d = a[i] - b[i]
            s += d * d
        return s
