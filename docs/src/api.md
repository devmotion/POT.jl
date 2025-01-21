# API

## Exact optimal transport

```@docs
emd
emd2
emd_1d
emd2_1d
```

## Regularized optimal transport

```@docs
sinkhorn
sinkhorn2
empirical_sinkhorn_divergence
barycenter
```

The submodule `Smooth` contains a function for solving regularized optimal
transport problems with L2- and entropic regularization using the dual
formulation. You can load the submodule with
```julia
using PythonOT.Smooth
```

```@docs
PythonOT.Smooth.smooth_ot_dual
```

## Unbalanced optimal transport

```@docs
sinkhorn_unbalanced
sinkhorn_unbalanced2
barycenter_unbalanced
mm_unbalanced
```

## Partial optimal transport

```@docs
entropic_partial_wasserstein
```
