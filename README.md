# k_dimensional_tree

Stores k-dimensional points in a binary tree for nearest-neighbor and orthogonal range queries. Pure Python, standard library only.

## Usage

```python
from k_dimensional_tree import KDTree

tree = KDTree([(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)])

# Closest point to a query
nearest = tree.nearest_neighbor((9, 2))   # (8, 1)

# All points within a bounding box
inside = tree.range_query((4, 2), (7, 7)) # [(5, 4), (4, 7), (7, 2)]
```

`Point` is also exported as an alias for `tuple`:

```python
from k_dimensional_tree import KDTree, Point
```

## Why

For small-to-medium point sets where importing a compiled spatial index is more trouble than it is worth. The trade-off is speed: this pure-Python tree is fine for a few thousand points; at hundreds of thousands you want something backed by C. Range queries return points in tree-traversal order, not sorted order—sort the result yourself if order matters.

## Edge cases

Duplicate points are stored, not collapsed. `nearest_neighbor` returns one of the equal-distance duplicates deterministically (left subtree preferred on ties). Points must be plain `tuple` values; lists are rejected with a `TypeError`. All points in a tree must share the same dimensionality.

## Running the tests

```
PYTHONPATH=src python -m unittest discover -s tests
```
