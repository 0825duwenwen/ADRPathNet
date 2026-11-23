"""Minimal dependency fallback for local structure-only preflight.

Never used when the production torch_scatter package is installed. The
preflight constructs modules but performs no graph propagation.
"""
from __future__ import annotations
import sys, types
import torch

def install_torch_scatter_fixture():
    module = types.ModuleType("torch_scatter")
    def scatter(src, index, dim=-2, dim_size=None, reduce="sum"):
        if dim < 0: dim += src.ndim
        shape = list(src.shape); shape[dim] = int(dim_size)
        output = torch.zeros(shape, dtype=src.dtype, device=src.device)
        return torch.index_add(output, dim, index, src)
    module.scatter = scatter
    module.scatter_add = lambda src, index, dim=0, dim_size=None: scatter(src, index, dim, dim_size, "add")
    sys.modules["torch_scatter"] = module
    try:
        import torch_geometric  # noqa: F401
        return
    except ImportError:
        pass
    geometric = types.ModuleType("torch_geometric")
    nn_module = types.ModuleType("torch_geometric.nn")
    conv_module = types.ModuleType("torch_geometric.nn.conv")
    utils_module = types.ModuleType("torch_geometric.utils")
    class MessagePassing(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.node_dim = kwargs.get("node_dim", -2)
            self.inspector = types.SimpleNamespace(params={"message": {"input_i": None}})
            self._user_args = ["input_j", "input_i", "relation", "boundary", "edge_type"]
    conv_module.MessagePassing = MessagePassing
    utils_module.degree = lambda index, num_nodes=None, dim_size=None: torch.bincount(
        index, minlength=(num_nodes or dim_size)).float()
    sys.modules.update({"torch_geometric": geometric, "torch_geometric.nn": nn_module,
                        "torch_geometric.nn.conv": conv_module, "torch_geometric.utils": utils_module})
