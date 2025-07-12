+++
date = '2025-07-12T14:14:51+08:00'
draft = true
title = 'DDG Course 2-4: Geometry and Mathematical Basis'
summary = 'Mesh; Manifold; Exterior Algebra; K-forms'
+++

# Mesh
## Simplex
- A k-simplex is a k-dimensional convex hull of its k+1 affinely independent points, which we call vertices.
- Simplex -> Barycentric coordinates

  - In the triangle shown with vertices $p_1, p_2, p_3$, the point $p$ can be described by the specific barycentric coordinates $(t_1, t_2, t_3) = (0.2, 0.4, 0.4)$:

$$
p = \frac{2}{10}p_1 + \frac{4}{10}p_2 + \frac{4}{10}p_3
$$

## Simplicial Complex
- A bunch of simplices
- Face definition 1: A face of a simplex is any lower-dimensional simplex formed by a subset of its vertices.
- A geometric simplicial complex: a collection of simplices where
  - The intersection of any two simplices is either empty or a simplex in the complex.
  - every face of a simplex in the complex is also in the complex.
- An abstract simplicial complex: for each simplex, all its subsets are also contained in the complex.
  - Only how things are connected matters, not their geometric embedding.

## Orientation
- For 2-simplex, specify by tuple (even permutation)

# Manifold

# Exterior Algebra

# K-forms
