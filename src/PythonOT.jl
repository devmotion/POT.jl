module PythonOT

using PyCall: PyCall

export emd,
    emd2,
    emd_1d,
    emd2_1d,
    sinkhorn,
    sinkhorn2,
    barycenter,
    barycenter_unbalanced,
    sinkhorn_unbalanced,
    sinkhorn_unbalanced2,
    empirical_sinkhorn_divergence,
    mm_unbalanced,
    entropic_partial_wasserstein

const pot = PyCall.PyNULL()

include("lib.jl")
include("smooth.jl")

function __init__()
    return copy!(pot, PyCall.pyimport_conda("ot", "pot", "conda-forge"))
end

end
