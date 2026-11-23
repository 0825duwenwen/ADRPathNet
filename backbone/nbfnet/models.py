import copy
from collections.abc import Sequence

import torch
from torch import nn, autograd

from torch_scatter import scatter_add
from . import tasks, layers
from .prototypes import SparseIngredientPrototypeRefiner


class NBFNet(nn.Module):

    def __init__(self, input_dim, hidden_dims, num_relation, message_func="distmult", aggregate_func="pna",
                 short_cut=False, layer_norm=False, activation="relu", concat_hidden=False, num_mlp_layer=2,
                 dependent=True, remove_one_hop=False, num_beam=10, path_topk=10,
                 use_relation_gate=False,
                 relation_gate_scale=0.1,
                 relation_gate_mode=None):
        super(NBFNet, self).__init__()

        if not isinstance(use_relation_gate, bool):
            raise ValueError("use_relation_gate must be a boolean")
        valid_gate_modes = ("static", "dynamic", "query_dynamic", "message_dynamic")
        if use_relation_gate and relation_gate_mode not in valid_gate_modes:
            raise ValueError(
                "relation_gate_mode must be one of %s when the gate is enabled"
                % (valid_gate_modes,)
            )
        if not use_relation_gate and relation_gate_mode not in (None, "none"):
            raise ValueError(
                "relation_gate_mode must be absent or exactly 'none' when the gate is disabled"
            )
        if use_relation_gate and relation_gate_scale <= 0:
            raise ValueError("relation_gate_scale must be greater than 0")
        if relation_gate_mode == "message_dynamic" and relation_gate_scale != 0.5:
            raise ValueError("NBFNet-DGate requires relation_gate_scale=0.5")

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]

        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.short_cut = short_cut  # whether to use residual connections between GNN layers
        self.concat_hidden = concat_hidden  # whether to compute final states as a function of all layer outputs or last
        self.remove_one_hop = remove_one_hop  # whether to dynamically remove one-hop edges from edge_index
        self.num_beam = num_beam
        self.path_topk = path_topk
        self.capture_gate = False

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layers.GeneralizedRelationalConv(self.dims[i], self.dims[i + 1], num_relation,
                                                                self.dims[0], message_func, aggregate_func, layer_norm,
                                                                activation, dependent,
                use_relation_gate=use_relation_gate,
                relation_gate_scale=relation_gate_scale,
                relation_gate_mode=relation_gate_mode,
                layer_id=i))

        feature_dim = (sum(hidden_dims) if concat_hidden else hidden_dims[-1]) + input_dim

        # additional relation embedding which serves as an initial 'query' for the NBFNet forward pass
        # each layer has its own learnable relations matrix, so we send the total number of relations, too
        self.query = nn.Embedding(num_relation, input_dim)
        self.mlp = nn.Sequential()
        mlp = []
        for i in range(num_mlp_layer - 1):
            mlp.append(nn.Linear(feature_dim, feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)

    def remove_easy_edges(self, data, h_index, t_index, r_index=None):
        # we remove training edges (we need to predict them at training time) from the edge index
        # think of it as a dynamic edge dropout
        h_index_ext = torch.cat([h_index, t_index], dim=-1)
        t_index_ext = torch.cat([t_index, h_index], dim=-1)
        r_index_ext = torch.cat([r_index, r_index + self.num_relation // 2], dim=-1)
        if self.remove_one_hop:
            # we remove all existing immediate edges between heads and tails in the batch
            edge_index = data.edge_index
            easy_edge = torch.stack([h_index_ext, t_index_ext]).flatten(1)
            index = tasks.edge_match(edge_index, easy_edge)[0]
            mask = ~index_to_mask(index, data.num_edges)
        else:
            # we remove existing immediate edges between heads and tails in the batch with the given relation
            edge_index = torch.cat([data.edge_index, data.edge_type.unsqueeze(0)])
            # note that here we add relation types r_index_ext to the matching query
            easy_edge = torch.stack([h_index_ext, t_index_ext, r_index_ext]).flatten(1)
            index = tasks.edge_match(edge_index, easy_edge)[0]
            mask = ~index_to_mask(index, data.num_edges)

        data = copy.copy(data)
        data.edge_index = data.edge_index[:, mask]
        data.edge_type = data.edge_type[mask]
        return data

    def negative_sample_to_tail(self, h_index, t_index, r_index):
        # convert p(h | t, r) to p(t' | h', r')
        # h' = t, r' = r^{-1}, t' = h
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index + self.num_relation // 2)
        return new_h_index, new_t_index, new_r_index

    def bellmanford(self, data, h_index, r_index, separate_grad=False):
        batch_size = len(r_index)

        # initialize queries (relation types of the given triples)
        query = self.query(r_index)
        index = h_index.unsqueeze(-1).expand_as(query)

        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        # by the scatter operation we put query (relation) embeddings as init features of source (index) nodes
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()
            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(
                layer_input,
                query,
                boundary,
                data.edge_index,
                data.edge_type,
                size,
                edge_weight,
                query_ids=h_index,
            )
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, data, batch):
        h_index, t_index, r_index = batch.unbind(-1)
        if self.training:
            # Edge dropout in the training mode
            # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths
            data = self.remove_easy_edges(data, h_index, t_index, r_index)

        shape = h_index.shape
        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        # message passing and updated node representations
        output = self.bellmanford(data, h_index[:, 0], r_index[:, 0])  # (num_nodes, batch_size, feature_dim）
        feature = output["node_feature"]
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        # extract representations of tail entities from the updated node states
        feature = feature.gather(1, index)  # (batch_size, num_negative + 1, feature_dim)

        # probability logit for each tail node in the batch
        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
        score = self.mlp(feature).squeeze(-1)
        return score.view(shape)

    def visualize(self, data, batch):
        assert batch.shape == (1, 3)
        h_index, t_index, r_index = batch.unbind(-1)

        output = self.bellmanford(data, h_index, r_index, separate_grad=True)
        feature = output["node_feature"]
        edge_weights = output["edge_weights"]

        index = t_index.unsqueeze(0).unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        feature = feature.gather(1, index).squeeze(0)
        score = self.mlp(feature).squeeze(-1)

        edge_grads = autograd.grad(score, edge_weights)
        distances, back_edges = self.beam_search_distance(data, edge_grads, h_index, t_index, self.num_beam)
        paths, weights = self.topk_average_length(distances, back_edges, t_index, self.path_topk)

        return paths, weights

    @torch.no_grad()
    def beam_search_distance(self, data, edge_grads, h_index, t_index, num_beam=10):
        # beam search the top-k distance from h to t (and to every other node)
        num_nodes = data.num_nodes
        input = torch.full((num_nodes, num_beam), float("-inf"), device=h_index.device)
        input[h_index, 0] = 0
        edge_mask = data.edge_index[0, :] != t_index

        distances = []
        back_edges = []
        for edge_grad in edge_grads:
            # we don't allow any path goes out of t once it arrives at t
            node_in, node_out = data.edge_index[:, edge_mask]
            relation = data.edge_type[edge_mask]
            edge_grad = edge_grad[edge_mask]

            message = input[node_in] + edge_grad.unsqueeze(-1) # (num_edges, num_beam)
            # (num_edges, num_beam, 3)
            msg_source = torch.stack([node_in, node_out, relation], dim=-1).unsqueeze(1).expand(-1, num_beam, -1)

            # (num_edges, num_beam)
            is_duplicate = torch.isclose(message.unsqueeze(-1), message.unsqueeze(-2)) & \
                           (msg_source.unsqueeze(-2) == msg_source.unsqueeze(-3)).all(dim=-1)
            # pick the first occurrence as the ranking in the previous node's beam
            # this makes deduplication easier later
            # and store it in msg_source
            is_duplicate = is_duplicate.float() - \
                           torch.arange(num_beam, dtype=torch.float, device=message.device) / (num_beam + 1)
            prev_rank = is_duplicate.argmax(dim=-1, keepdim=True)
            msg_source = torch.cat([msg_source, prev_rank], dim=-1) # (num_edges, num_beam, 4)

            node_out, order = node_out.sort()
            node_out_set = torch.unique(node_out)
            # sort messages w.r.t. node_out
            message = message[order].flatten() # (num_edges * num_beam)
            msg_source = msg_source[order].flatten(0, -2) # (num_edges * num_beam, 4)
            size = node_out.bincount(minlength=num_nodes)
            msg2out = size_to_index(size[node_out_set] * num_beam)
            # deduplicate messages that are from the same source and the same beam
            is_duplicate = (msg_source[1:] == msg_source[:-1]).all(dim=-1)
            is_duplicate = torch.cat([torch.zeros(1, dtype=torch.bool, device=message.device), is_duplicate])
            message = message[~is_duplicate]
            msg_source = msg_source[~is_duplicate]
            msg2out = msg2out[~is_duplicate]
            size = msg2out.bincount(minlength=len(node_out_set))

            if not torch.isinf(message).all():
                # take the topk messages from the neighborhood
                # distance: (len(node_out_set) * num_beam)
                distance, rel_index = scatter_topk(message, size, k=num_beam)
                abs_index = rel_index + (size.cumsum(0) - size).unsqueeze(-1)
                # store msg_source for backtracking
                back_edge = msg_source[abs_index] # (len(node_out_set) * num_beam, 4)
                distance = distance.view(len(node_out_set), num_beam)
                back_edge = back_edge.view(len(node_out_set), num_beam, 4)
                # scatter distance / back_edge back to all nodes
                distance = scatter_add(distance, node_out_set, dim=0, dim_size=num_nodes) # (num_nodes, num_beam)
                back_edge = scatter_add(back_edge, node_out_set, dim=0, dim_size=num_nodes) # (num_nodes, num_beam, 4)
            else:
                distance = torch.full((num_nodes, num_beam), float("-inf"), device=message.device)
                back_edge = torch.zeros(num_nodes, num_beam, 4, dtype=torch.long, device=message.device)

            distances.append(distance)
            back_edges.append(back_edge)
            input = distance

        return distances, back_edges

    def topk_average_length(self, distances, back_edges, t_index, k=10):
        # backtrack distances and back_edges to generate the paths
        paths = []
        average_lengths = []

        for i in range(len(distances)):
            distance, order = distances[i][t_index].flatten(0, -1).sort(descending=True)
            back_edge = back_edges[i][t_index].flatten(0, -2)[order]
            for d, (h, t, r, prev_rank) in zip(distance[:k].tolist(), back_edge[:k].tolist()):
                if d == float("-inf"):
                    break
                path = [(h, t, r)]
                for j in range(i - 1, -1, -1):
                    h, t, r, prev_rank = back_edges[j][h, prev_rank].tolist()
                    path.append((h, t, r))
                paths.append(path[::-1])
                average_lengths.append(d / len(path))

        if paths:
            average_lengths, paths = zip(*sorted(zip(average_lengths, paths), reverse=True)[:k])

        return paths, average_lengths

    def relation_gate_values(self):
        values = []
        for layer in self.layers:
            if not hasattr(layer, "relation_gate_values"):
                continue
            layer_values = layer.relation_gate_values()
            if layer_values is not None:
                values.append(layer_values)
        if not values:
            return None
        return values

    def enable_dgate_capture(self, collector):
        dgate_layers = [
            layer
            for layer in self.layers
            if layer.relation_gate_mode == "message_dynamic"
        ]
        if not dgate_layers:
            raise RuntimeError("Gate capture is only available for NBFNet-DGate")
        for layer in dgate_layers:
            layer.gate_capture = collector
        self.capture_gate = True

    def disable_dgate_capture(self):
        for layer in self.layers:
            layer.gate_capture = None
        self.capture_gate = False


class NBFNetSIPR(NBFNet):
    """NBFNet with sparse-evidence ingredient prototype refinement."""

    def __init__(
        self,
        train_data,
        prototype_dim=8,
        prototype_scale=0.1,
        prototype_scale_max=1.0,
        contains_relation_id=0,
        target_relation_id=2,
        **kwargs,
    ):
        if kwargs.get("use_relation_gate", False):
            raise ValueError(
                "NBFNet-SIPR cannot be combined with a relation gate"
            )

        relation_gate_mode = kwargs.get("relation_gate_mode")
        if relation_gate_mode not in (None, "none"):
            raise ValueError(
                "NBFNet-SIPR requires relation_gate_mode to be absent "
                "or exactly 'none'"
            )

        super().__init__(**kwargs)

        self.prototype_refiner = SparseIngredientPrototypeRefiner(
            train_data=train_data,
            num_relation=self.num_relation,
            prototype_dim=prototype_dim,
            prototype_scale=prototype_scale,
            prototype_scale_max=prototype_scale_max,
            contains_relation_id=contains_relation_id,
            target_relation_id=target_relation_id,
        )

    def forward(self, data, batch):
        base_score = super().forward(data, batch)
        prototype_correction = self.prototype_refiner(batch)
        return base_score + prototype_correction

    def prototype_identity(self):
        return self.prototype_refiner.identity()

    def prototype_diagnostics(self, batch):
        return self.prototype_refiner(
            batch,
            return_diagnostics=True,
        )[1]



class NBFNetEviMoE(NBFNet):
    """
    NBFNet with candidate-list evidence mixture.

    Target-relation candidate lists are scored by a probability mixture
    of the NBFNet path expert, exact ingredient evidence expert,
    multi-prototype ingredient expert, and organ evidence expert.
    Other relations preserve the original NBFNet logits.
    """

    def __init__(
        self,
        train_data,
        ingredient_num_slots=4,
        ingredient_slot_dim=8,
        ingredient_slot_temperature=0.5,
        evidence_router_hidden_dim=16,
        evidence_base_prior=0.85,
        contains_relation_id=0,
        affects_relation_id=1,
        target_relation_id=2,
        **kwargs,
    ):
        from .evidence_moe import (
            CandidateEvidenceMixture,
            MultiPrototypeIngredientExpert,
            OrganEvidenceExpert,
        )

        if kwargs.get("use_relation_gate", False):
            raise ValueError(
                "NBFNet-EviMoE cannot be combined "
                "with a relation gate"
            )

        relation_gate_mode = kwargs.get(
            "relation_gate_mode"
        )
        if relation_gate_mode not in (None, "none"):
            raise ValueError(
                "NBFNet-EviMoE requires "
                "relation_gate_mode to be absent "
                "or exactly 'none'"
            )

        super().__init__(**kwargs)

        self.target_relation_id = int(
            target_relation_id
        )

        self.ingredient_expert = (
            MultiPrototypeIngredientExpert(
                train_data=train_data,
                num_relation=self.num_relation,
                num_slots=ingredient_num_slots,
                slot_dim=ingredient_slot_dim,
                contains_relation_id=
                    contains_relation_id,
                target_relation_id=
                    target_relation_id,
                slot_temperature=
                    ingredient_slot_temperature,
            )
        )
        self.organ_expert = OrganEvidenceExpert(
            train_data=train_data,
            num_relation=self.num_relation,
            affects_relation_id=affects_relation_id,
            target_relation_id=target_relation_id,
        )
        self.evidence_mixture = (
            CandidateEvidenceMixture(
                num_evidence_experts=3,
                router_feature_dim=9,
                router_hidden_dim=
                    evidence_router_hidden_dim,
                base_prior=evidence_base_prior,
            )
        )

        self.uses_candidate_list_loss = True

    def forward(self, data, batch):
        base_logits = super().forward(
            data,
            batch,
        )

        if base_logits.ndim != 2:
            raise ValueError(
                "NBFNet-EviMoE requires candidate "
                "lists with shape [batch, candidates]"
            )

        (
            ingredient_evidence_logits,
            ingredient_router_features,
        ) = self.ingredient_expert(batch)
        (
            organ_evidence_logits,
            organ_router_features,
        ) = self.organ_expert(batch)

        evidence_logits = torch.cat(
            [
                ingredient_evidence_logits,
                organ_evidence_logits.unsqueeze(-1),
            ],
            dim=-1,
        )
        router_features = torch.cat(
            [
                ingredient_router_features,
                organ_router_features,
            ],
            dim=-1,
        )

        mixture_log_probabilities = (
            self.evidence_mixture(
                base_logits,
                evidence_logits,
                router_features,
            )
        )

        relation_index = batch[..., 2]
        target_rows = (
            relation_index
            == self.target_relation_id
        ).all(dim=-1)

        output = base_logits.clone()
        output[target_rows] = (
            mixture_log_probabilities[target_rows]
        )
        return output

    def load_pretrained_backbone_state(
        self,
        backbone_state,
    ):
        if not isinstance(backbone_state, dict):
            raise TypeError(
                "backbone_state must be a state dictionary"
            )

        evidence_prefixes = (
            "ingredient_expert.",
            "organ_expert.",
            "evidence_mixture.",
        )
        current_state = self.state_dict()
        expected_keys = {
            key
            for key in current_state
            if not key.startswith(evidence_prefixes)
        }
        provided_keys = set(backbone_state)

        missing = sorted(
            expected_keys - provided_keys
        )
        unexpected = sorted(
            provided_keys - expected_keys
        )
        shape_mismatches = {
            key: {
                "expected": list(
                    current_state[key].shape
                ),
                "provided": list(
                    backbone_state[key].shape
                ),
            }
            for key in sorted(
                expected_keys & provided_keys
            )
            if tuple(current_state[key].shape)
            != tuple(backbone_state[key].shape)
        }

        if missing or unexpected or shape_mismatches:
            raise ValueError(
                "Pretrained NBFNet backbone is not "
                "strictly compatible: missing=%r, "
                "unexpected=%r, shape_mismatches=%r"
                % (
                    missing,
                    unexpected,
                    shape_mismatches,
                )
            )

        updated_state = dict(current_state)
        for key in expected_keys:
            updated_state[key] = backbone_state[key]

        self.load_state_dict(
            updated_state,
            strict=True,
        )

        return {
            "loaded_key_count": len(expected_keys),
            "missing": missing,
            "unexpected": unexpected,
            "shape_mismatches": shape_mismatches,
        }

    def freeze_backbone(self):
        evidence_prefixes = (
            "ingredient_expert.",
            "organ_expert.",
            "evidence_mixture.",
        )
        frozen_tensor_count = 0
        frozen_parameter_count = 0
        trainable_tensor_count = 0
        trainable_parameter_count = 0

        for name, parameter in self.named_parameters():
            if name.startswith(evidence_prefixes):
                parameter.requires_grad_(True)
                trainable_tensor_count += 1
                trainable_parameter_count += (
                    parameter.numel()
                )
            else:
                parameter.requires_grad_(False)
                frozen_tensor_count += 1
                frozen_parameter_count += (
                    parameter.numel()
                )

        return {
            "frozen_tensor_count":
                frozen_tensor_count,
            "frozen_parameter_count":
                frozen_parameter_count,
            "trainable_tensor_count":
                trainable_tensor_count,
            "trainable_parameter_count":
                trainable_parameter_count,
        }

    def evidence_identity(self):
        return {
            "model_class": "NBFNetEviMoE",
            "model_display_name":
                "NBFNet-EviMoE",
            "target_relation_id":
                self.target_relation_id,
            "uses_candidate_list_loss":
                self.uses_candidate_list_loss,
            "ingredient_expert":
                self.ingredient_expert.identity(),
            "organ_expert":
                self.organ_expert.identity(),
            "mixture":
                self.evidence_mixture.identity(),
        }

    def evidence_diagnostics(self, batch):
        (
            ingredient_evidence_logits,
            ingredient_router_features,
            ingredient_diagnostics,
        ) = self.ingredient_expert(
            batch,
            return_diagnostics=True,
        )
        (
            organ_evidence_logits,
            organ_router_features,
            organ_diagnostics,
        ) = self.organ_expert(
            batch,
            return_diagnostics=True,
        )

        evidence_logits = torch.cat(
            [
                ingredient_evidence_logits,
                organ_evidence_logits.unsqueeze(-1),
            ],
            dim=-1,
        )
        router_features = torch.cat(
            [
                ingredient_router_features,
                organ_router_features,
            ],
            dim=-1,
        )

        return {
            "evidence_logits": evidence_logits,
            "router_features": router_features,
            "ingredient":
                ingredient_diagnostics,
            "organ": organ_diagnostics,
        }


class NBFNetEviRerank(NBFNet):
    """
    Frozen-backbone NBFNet with candidate-independent evidence reranking.

    Evidence corrections are computed independently for each candidate.
    Adding unrelated candidates therefore cannot change the correction
    assigned to an existing candidate. Zero initialization preserves the
    pretrained NBFNet ranking exactly.
    """

    def __init__(
        self,
        train_data,
        ingredient_num_slots=4,
        ingredient_slot_dim=8,
        ingredient_slot_temperature=0.5,
        reranker_hidden_dim=32,
        maximum_correction=1.0,
        contains_relation_id=0,
        affects_relation_id=1,
        target_relation_id=2,
        **kwargs,
    ):
        from .evidence_moe import (
            MultiPrototypeIngredientExpert,
            OrganEvidenceExpert,
            SetInvariantEvidenceReranker,
        )

        if kwargs.get("use_relation_gate", False):
            raise ValueError(
                "NBFNet-EviRerank cannot be combined "
                "with a relation gate"
            )

        relation_gate_mode = kwargs.get(
            "relation_gate_mode"
        )
        if relation_gate_mode not in (None, "none"):
            raise ValueError(
                "NBFNet-EviRerank requires "
                "relation_gate_mode to be absent "
                "or exactly 'none'"
            )

        super().__init__(**kwargs)

        self.target_relation_id = int(
            target_relation_id
        )

        self.ingredient_expert = (
            MultiPrototypeIngredientExpert(
                train_data=train_data,
                num_relation=self.num_relation,
                num_slots=ingredient_num_slots,
                slot_dim=ingredient_slot_dim,
                contains_relation_id=
                    contains_relation_id,
                target_relation_id=
                    target_relation_id,
                slot_temperature=
                    ingredient_slot_temperature,
            )
        )
        self.organ_expert = OrganEvidenceExpert(
            train_data=train_data,
            num_relation=self.num_relation,
            affects_relation_id=affects_relation_id,
            target_relation_id=target_relation_id,
        )
        self.evidence_reranker = (
            SetInvariantEvidenceReranker(
                num_evidence_experts=3,
                router_feature_dim=9,
                hidden_dim=reranker_hidden_dim,
                maximum_correction=
                    maximum_correction,
            )
        )

        self.uses_candidate_list_loss = False
        self.uses_pairwise_rerank_loss = True

    def _evidence_outputs(
        self,
        batch,
        return_diagnostics=False,
    ):
        if return_diagnostics:
            (
                ingredient_evidence,
                ingredient_router,
                ingredient_diagnostics,
            ) = self.ingredient_expert(
                batch,
                return_diagnostics=True,
            )
            (
                organ_evidence,
                organ_router,
                organ_diagnostics,
            ) = self.organ_expert(
                batch,
                return_diagnostics=True,
            )
        else:
            (
                ingredient_evidence,
                ingredient_router,
            ) = self.ingredient_expert(batch)
            (
                organ_evidence,
                organ_router,
            ) = self.organ_expert(batch)

        evidence_logits = torch.cat(
            [
                ingredient_evidence,
                organ_evidence.unsqueeze(-1),
            ],
            dim=-1,
        )
        router_features = torch.cat(
            [
                ingredient_router,
                organ_router,
            ],
            dim=-1,
        )

        if not return_diagnostics:
            return evidence_logits, router_features

        return (
            evidence_logits,
            router_features,
            ingredient_diagnostics,
            organ_diagnostics,
        )

    def forward(self, data, batch):
        base_logits = super().forward(
            data,
            batch,
        )

        if base_logits.ndim != 2:
            raise ValueError(
                "NBFNet-EviRerank requires candidate "
                "lists with shape [batch, candidates]"
            )

        evidence_logits, router_features = (
            self._evidence_outputs(batch)
        )
        reranked_logits = self.evidence_reranker(
            base_logits,
            evidence_logits,
            router_features,
        )

        relation_index = batch[..., 2]
        target_rows = (
            relation_index
            == self.target_relation_id
        ).all(dim=-1)

        output = base_logits.clone()
        output[target_rows] = (
            reranked_logits[target_rows]
        )
        return output

    def load_pretrained_backbone_state(
        self,
        backbone_state,
    ):
        if not isinstance(backbone_state, dict):
            raise TypeError(
                "backbone_state must be a state dictionary"
            )

        evidence_prefixes = (
            "ingredient_expert.",
            "organ_expert.",
            "evidence_reranker.",
        )
        current_state = self.state_dict()
        expected_keys = {
            key
            for key in current_state
            if not key.startswith(evidence_prefixes)
        }
        provided_keys = set(backbone_state)

        missing = sorted(
            expected_keys - provided_keys
        )
        unexpected = sorted(
            provided_keys - expected_keys
        )
        shape_mismatches = {
            key: {
                "expected": list(
                    current_state[key].shape
                ),
                "provided": list(
                    backbone_state[key].shape
                ),
            }
            for key in sorted(
                expected_keys & provided_keys
            )
            if tuple(current_state[key].shape)
            != tuple(backbone_state[key].shape)
        }

        if missing or unexpected or shape_mismatches:
            raise ValueError(
                "Pretrained NBFNet backbone is not "
                "strictly compatible: missing=%r, "
                "unexpected=%r, shape_mismatches=%r"
                % (
                    missing,
                    unexpected,
                    shape_mismatches,
                )
            )

        updated_state = dict(current_state)
        for key in expected_keys:
            updated_state[key] = backbone_state[key]

        self.load_state_dict(
            updated_state,
            strict=True,
        )

        return {
            "loaded_key_count": len(expected_keys),
            "missing": missing,
            "unexpected": unexpected,
            "shape_mismatches": shape_mismatches,
        }

    def freeze_backbone(self):
        evidence_prefixes = (
            "ingredient_expert.",
            "organ_expert.",
            "evidence_reranker.",
        )
        frozen_tensor_count = 0
        frozen_parameter_count = 0
        trainable_tensor_count = 0
        trainable_parameter_count = 0

        for name, parameter in self.named_parameters():
            if name.startswith(evidence_prefixes):
                parameter.requires_grad_(True)
                trainable_tensor_count += 1
                trainable_parameter_count += (
                    parameter.numel()
                )
            else:
                parameter.requires_grad_(False)
                frozen_tensor_count += 1
                frozen_parameter_count += (
                    parameter.numel()
                )

        return {
            "frozen_tensor_count":
                frozen_tensor_count,
            "frozen_parameter_count":
                frozen_parameter_count,
            "trainable_tensor_count":
                trainable_tensor_count,
            "trainable_parameter_count":
                trainable_parameter_count,
        }

    def evidence_identity(self):
        return {
            "model_class": "NBFNetEviRerank",
            "model_display_name":
                "NBFNet-EviRerank",
            "target_relation_id":
                self.target_relation_id,
            "uses_candidate_list_loss":
                self.uses_candidate_list_loss,
            "uses_pairwise_rerank_loss":
                self.uses_pairwise_rerank_loss,
            "ingredient_expert":
                self.ingredient_expert.identity(),
            "organ_expert":
                self.organ_expert.identity(),
            "reranker":
                self.evidence_reranker.identity(),
        }

    def evidence_diagnostics(
        self,
        batch,
        base_logits=None,
    ):
        (
            evidence_logits,
            router_features,
            ingredient_diagnostics,
            organ_diagnostics,
        ) = self._evidence_outputs(
            batch,
            return_diagnostics=True,
        )

        result = {
            "evidence_logits": evidence_logits,
            "router_features": router_features,
            "ingredient":
                ingredient_diagnostics,
            "organ": organ_diagnostics,
        }

        if base_logits is not None:
            (
                reranked_logits,
                reranker_diagnostics,
            ) = self.evidence_reranker(
                base_logits,
                evidence_logits,
                router_features,
                return_diagnostics=True,
            )
            result["reranked_logits"] = (
                reranked_logits
            )
            result["reranker"] = (
                reranker_diagnostics
            )

        return result


def index_to_mask(index, size):
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask


def size_to_index(size):
    range = torch.arange(len(size), device=size.device)
    index2sample = range.repeat_interleave(size)
    return index2sample


def multi_slice_mask(starts, ends, length):
    values = torch.cat([torch.ones_like(starts), -torch.ones_like(ends)])
    slices = torch.cat([starts, ends])
    mask = scatter_add(values, slices, dim=0, dim_size=length + 1)[:-1]
    mask = mask.cumsum(0).bool()
    return mask


def scatter_extend(data, size, input, input_size):
    new_size = size + input_size
    new_cum_size = new_size.cumsum(0)
    new_data = torch.zeros(new_cum_size[-1], *data.shape[1:], dtype=data.dtype, device=data.device)
    starts = new_cum_size - new_size
    ends = starts + size
    index = multi_slice_mask(starts, ends, new_cum_size[-1])
    new_data[index] = data
    new_data[~index] = input
    return new_data, new_size


def scatter_topk(input, size, k, largest=True):
    index2graph = size_to_index(size)
    index2graph = index2graph.view([-1] + [1] * (input.ndim - 1))

    mask = ~torch.isinf(input)
    max = input[mask].max().item()
    min = input[mask].min().item()
    safe_input = input.clamp(2 * min - max, 2 * max - min)
    offset = (max - min) * 4
    if largest:
        offset = -offset
    input_ext = safe_input + offset * index2graph
    index_ext = input_ext.argsort(dim=0, descending=largest)
    num_actual = size.clamp(max=k)
    num_padding = k - num_actual
    starts = size.cumsum(0) - size
    ends = starts + num_actual
    mask = multi_slice_mask(starts, ends, len(index_ext)).nonzero().flatten()

    if (num_padding > 0).any():
        # special case: size < k, pad with the last valid index
        padding = ends - 1
        padding2graph = size_to_index(num_padding)
        mask = scatter_extend(mask, num_actual, padding[padding2graph], num_padding)[0]

    index = index_ext[mask] # (N * k, ...)
    value = input.gather(0, index)
    if isinstance(k, torch.Tensor) and k.shape == size.shape:
        value = value.view(-1, *input.shape[1:])
        index = index.view(-1, *input.shape[1:])
        index = index - (size.cumsum(0) - size).repeat_interleave(k).view([-1] + [1] * (index.ndim - 1))
    else:
        value = value.view(-1, k, *input.shape[1:])
        index = index.view(-1, k, *input.shape[1:])
        index = index - (size.cumsum(0) - size).view([-1] + [1] * (index.ndim - 1))

    return value, index
