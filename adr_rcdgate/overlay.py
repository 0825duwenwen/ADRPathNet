from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch

from .gate import ReceiverCompatibilityDynamicGate


def model_kwargs(gated: bool, model_config):
    return dict(
        input_dim=model_config.input_dim,
        hidden_dims=list(model_config.hidden_dims),
        num_relation=model_config.num_relation,
        message_func=model_config.message_func,
        aggregate_func=model_config.aggregate_func,
        short_cut=model_config.short_cut,
        layer_norm=model_config.layer_norm,
        dependent=model_config.dependent,
        use_relation_gate=gated,
        relation_gate_mode="message_dynamic" if gated else None,
        relation_gate_scale=model_config.relation_gate_scale,
    )


def install_rcdgate_overlay(nbfnet_root):
    """Patch the production layer class in memory; no production file is changed."""
    root = str(Path(nbfnet_root).resolve())
    script = str(Path(root) / "script")
    for item in (root, script):
        if item not in sys.path:
            sys.path.insert(0, item)
    production_layers = importlib.import_module("nbfnet.layers")
    layer_class = production_layers.GeneralizedRelationalConv
    if not hasattr(layer_class, "_adr_rcdgate_original_init"):
        layer_class._adr_rcdgate_original_init = layer_class.__init__
        layer_class._adr_rcdgate_original_message = layer_class.message
        layer_class._adr_rcdgate_original_message_and_aggregate = layer_class.message_and_aggregate

        def rcd_init(self, *args, **kwargs):
            requested = bool(kwargs.get("use_relation_gate", False))
            mode = kwargs.get("relation_gate_mode")
            scale = kwargs.get("relation_gate_scale", 0.1)
            if requested and mode == "message_dynamic":
                base_kwargs = dict(kwargs)
                base_kwargs.update(use_relation_gate=False, relation_gate_mode=None)
                layer_class._adr_rcdgate_original_init(self, *args, **base_kwargs)
                if "message" not in self.inspector.params:
                    raise RuntimeError("PyG Inspector did not register RCDGate message")
                if "input_i" not in self.inspector.params["message"]:
                    raise RuntimeError("PyG Inspector lost receiver argument input_i")
                if "input_i" not in self._user_args:
                    raise RuntimeError("PyG MessagePassing _user_args lost input_i")
                self.use_relation_gate = True
                self.relation_gate_mode = "message_dynamic"
                self.relation_gate_scale = scale
                self.dynamic_message_relation_gate = ReceiverCompatibilityDynamicGate(
                    self.input_dim, gate_scale=scale
                )
            else:
                layer_class._adr_rcdgate_original_init(self, *args, **kwargs)

        def rcd_message(self, input_j, input_i, relation, boundary, edge_type):
            relation_j = relation.index_select(self.node_dim, edge_type)
            if self.message_func == "transe":
                message = input_j + relation_j
            elif self.message_func == "distmult":
                message = input_j * relation_j
            elif self.message_func == "rotate":
                x_j_re, x_j_im = input_j.chunk(2, dim=-1)
                r_j_re, r_j_im = relation_j.chunk(2, dim=-1)
                message = torch.cat(
                    [x_j_re * r_j_re - x_j_im * r_j_im,
                     x_j_re * r_j_im + x_j_im * r_j_re], dim=-1
                )
            else:
                raise ValueError("Unknown message function `%s`" % self.message_func)

            if self.use_relation_gate and self.relation_gate_mode == "message_dynamic":
                if self.gate_capture is None:
                    message = self.dynamic_message_relation_gate(message, relation_j, input_i)
                else:
                    raw = message
                    message, logit, gate = self.dynamic_message_relation_gate.forward_with_details(
                        raw, relation_j, input_i
                    )
                    query_ids, edge_index, capture_edge_type = self._active_gate_context
                    self.gate_capture.capture(
                        self.layer_id, query_ids, edge_index, capture_edge_type,
                        raw, logit, gate,
                    )
            elif self.use_relation_gate:
                # RCDGate construction never enters legacy modes.
                raise RuntimeError("RCDGate overlay received an unsupported gate mode")
            return torch.cat([message, boundary], dim=self.node_dim)

        def rcd_message_and_aggregate(self, *args, **kwargs):
            if self.use_relation_gate and self.relation_gate_mode == "message_dynamic":
                raise RuntimeError(
                    "RCDGate requires explicit receiver-indexed messages; fused propagation is disabled"
                )
            return layer_class._adr_rcdgate_original_message_and_aggregate(self, *args, **kwargs)

        # PyG 2.4 Inspector keys its registry by callable.__name__. Keep the
        # real signatures above (especially input_i) and only restore the
        # canonical MessagePassing method identities. No functools.wraps is
        # used because it would expose the old message signature.
        rcd_init.__name__ = "__init__"
        rcd_message.__name__ = "message"
        rcd_message_and_aggregate.__name__ = "message_and_aggregate"
        layer_class.__init__ = rcd_init
        layer_class.message = rcd_message
        layer_class.message_and_aggregate = rcd_message_and_aggregate
    return importlib.import_module("nbfnet.models").NBFNet


def construct_matched_pair(nbfnet_root, model_config, seed=42):
    NBFNet = install_rcdgate_overlay(nbfnet_root)
    torch.manual_seed(seed)
    baseline = NBFNet(**model_kwargs(False, model_config))
    torch.manual_seed(seed)
    rcdgate = NBFNet(**model_kwargs(True, model_config))
    return baseline, rcdgate


def construct_model(nbfnet_root, gated: bool, model_config):
    NBFNet = install_rcdgate_overlay(nbfnet_root)
    kwargs = model_kwargs(gated, model_config)
    try:
        return NBFNet(**kwargs)
    except ValueError as error:
        # The supplied production constructor predates configurable RCDGate
        # scale and rejects message_dynamic values other than 0.5. Adapt only
        # in memory, then set the actual overlay gates to the requested scale.
        if not gated or model_config.relation_gate_scale == 0.5 or "requires relation_gate_scale=0.5" not in str(error):
            raise
        compatibility_kwargs = dict(kwargs, relation_gate_scale=0.5)
        model = NBFNet(**compatibility_kwargs)
        for layer in model.layers:
            layer.relation_gate_scale = model_config.relation_gate_scale
            layer.dynamic_message_relation_gate.gate_scale = model_config.relation_gate_scale
        return model
