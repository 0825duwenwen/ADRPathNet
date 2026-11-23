import math

import torch
from torch import nn
from torch.nn import functional as F


class CandidateEvidenceMixture(nn.Module):
    """
    Candidate-list mixture of a base path expert and evidence experts.

    Each expert first defines a probability distribution over the candidates.
    A candidate-level router then combines these calibrated distributions.
    This avoids directly adding structurally unrelated residuals to raw NBFNet
    logits, which was the main limitation of SIPR.
    """

    def __init__(
        self,
        num_evidence_experts,
        router_feature_dim,
        router_hidden_dim=16,
        base_prior=0.85,
        temperature_floor=0.05,
    ):
        super().__init__()

        if num_evidence_experts < 1:
            raise ValueError(
                "num_evidence_experts must be at least 1"
            )
        if router_feature_dim < 1:
            raise ValueError(
                "router_feature_dim must be at least 1"
            )
        if router_hidden_dim < 1:
            raise ValueError(
                "router_hidden_dim must be at least 1"
            )
        if not 0.0 < base_prior < 1.0:
            raise ValueError(
                "base_prior must be strictly between 0 and 1"
            )
        if temperature_floor <= 0.0:
            raise ValueError(
                "temperature_floor must be greater than 0"
            )

        self.num_evidence_experts = int(
            num_evidence_experts
        )
        self.num_experts = self.num_evidence_experts + 1
        self.router_feature_dim = int(router_feature_dim)
        self.router_hidden_dim = int(router_hidden_dim)
        self.base_prior = float(base_prior)
        self.temperature_floor = float(
            temperature_floor
        )

        self.router = nn.Sequential(
            nn.Linear(
                self.router_feature_dim,
                self.router_hidden_dim,
            ),
            nn.LayerNorm(self.router_hidden_dim),
            nn.GELU(),
            nn.Linear(
                self.router_hidden_dim,
                self.num_experts,
            ),
        )

        initial_raw_temperature = math.log(
            math.expm1(1.0 - self.temperature_floor)
        )
        self.raw_temperature = nn.Parameter(
            torch.full(
                (self.num_experts,),
                initial_raw_temperature,
                dtype=torch.float,
            )
        )

        self._initialize_router()

    def _initialize_router(self):
        final_layer = self.router[-1]

        nn.init.zeros_(final_layer.weight)

        evidence_prior = (
            1.0 - self.base_prior
        ) / self.num_evidence_experts
        priors = [
            self.base_prior,
            *(
                [evidence_prior]
                * self.num_evidence_experts
            ),
        ]

        with torch.no_grad():
            final_layer.bias.copy_(
                torch.tensor(priors).log()
            )

    def temperatures(self):
        return (
            self.temperature_floor
            + F.softplus(self.raw_temperature)
        )

    def forward(
        self,
        base_logits,
        evidence_logits,
        router_features,
        return_diagnostics=False,
    ):
        if base_logits.ndim != 2:
            raise ValueError(
                "base_logits must have shape "
                "[batch_size, num_candidates]"
            )

        expected_evidence_shape = (
            *base_logits.shape,
            self.num_evidence_experts,
        )
        if tuple(evidence_logits.shape) != (
            expected_evidence_shape
        ):
            raise ValueError(
                "evidence_logits must have shape %r, got %r"
                % (
                    expected_evidence_shape,
                    tuple(evidence_logits.shape),
                )
            )

        expected_router_shape = (
            *base_logits.shape,
            self.router_feature_dim,
        )
        if tuple(router_features.shape) != (
            expected_router_shape
        ):
            raise ValueError(
                "router_features must have shape %r, got %r"
                % (
                    expected_router_shape,
                    tuple(router_features.shape),
                )
            )

        expert_logits = torch.cat(
            [
                base_logits.unsqueeze(-1),
                evidence_logits,
            ],
            dim=-1,
        )

        temperatures = self.temperatures()
        calibrated_logits = (
            expert_logits
            / temperatures.view(1, 1, -1)
        )

        expert_log_probabilities = F.log_softmax(
            calibrated_logits,
            dim=1,
        )
        router_log_weights = F.log_softmax(
            self.router(router_features),
            dim=-1,
        )

        mixture_log_mass = torch.logsumexp(
            router_log_weights
            + expert_log_probabilities,
            dim=-1,
        )

        mixture_log_probabilities = (
            mixture_log_mass
            - torch.logsumexp(
                mixture_log_mass,
                dim=1,
                keepdim=True,
            )
        )

        if not return_diagnostics:
            return mixture_log_probabilities

        diagnostics = {
            "expert_log_probabilities":
                expert_log_probabilities,
            "router_weights":
                router_log_weights.exp(),
            "temperatures": temperatures,
        }
        return mixture_log_probabilities, diagnostics

    def identity(self):
        return {
            "fusion_mode":
                "candidate_probability_mixture",
            "num_experts": self.num_experts,
            "num_evidence_experts":
                self.num_evidence_experts,
            "router_feature_dim":
                self.router_feature_dim,
            "router_hidden_dim":
                self.router_hidden_dim,
            "base_prior": self.base_prior,
            "temperature_floor":
                self.temperature_floor,
        }



class MultiPrototypeIngredientExpert(nn.Module):
    """
    Training-graph-only ingredient evidence expert.

    A candidate ADR is represented by multiple learnable ingredient
    sub-prototypes rather than one global mean. Ingredient IDF weights
    reduce the influence of ubiquitous ingredients. Known training cause
    pairs use leave-one-drug-out candidate prototypes.
    """

    def __init__(
        self,
        train_data,
        num_relation,
        num_slots=4,
        slot_dim=8,
        contains_relation_id=0,
        target_relation_id=2,
        slot_temperature=0.5,
    ):
        super().__init__()

        from collections import Counter, defaultdict

        if num_relation < 2:
            raise ValueError(
                "num_relation must include inverse relations"
            )
        if num_slots < 2:
            raise ValueError(
                "num_slots must be at least 2"
            )
        if slot_dim < 1:
            raise ValueError(
                "slot_dim must be at least 1"
            )
        if slot_temperature <= 0:
            raise ValueError(
                "slot_temperature must be greater than 0"
            )

        self.num_node = int(train_data.num_nodes)
        self.num_relation = int(num_relation)
        self.num_slots = int(num_slots)
        self.slot_dim = int(slot_dim)
        self.contains_relation_id = int(
            contains_relation_id
        )
        self.target_relation_id = int(
            target_relation_id
        )
        self.slot_temperature = float(
            slot_temperature
        )

        edge_index = train_data.edge_index.detach().cpu()
        edge_type = train_data.edge_type.detach().cpu()

        contains_edges = edge_index[
            :, edge_type == self.contains_relation_id
        ].t().tolist()
        cause_edges = edge_index[
            :, edge_type == self.target_relation_id
        ].t().tolist()

        drug_to_ingredients = defaultdict(set)
        ingredient_to_drugs = defaultdict(set)
        drug_to_adrs = defaultdict(set)
        adr_to_drugs = defaultdict(set)

        for drug, ingredient in contains_edges:
            drug = int(drug)
            ingredient = int(ingredient)
            drug_to_ingredients[drug].add(ingredient)
            ingredient_to_drugs[ingredient].add(drug)

        for drug, adr in cause_edges:
            drug = int(drug)
            adr = int(adr)
            drug_to_adrs[drug].add(adr)
            adr_to_drugs[adr].add(drug)

        ingredient_ids = sorted(ingredient_to_drugs)
        if not ingredient_ids:
            raise ValueError(
                "training graph contains no ingredients"
            )

        ingredient_to_local = {
            entity_id: local_id
            for local_id, entity_id in enumerate(
                ingredient_ids
            )
        }

        num_ingredient_drugs = max(
            1,
            len(drug_to_ingredients),
        )
        ingredient_idf = []

        for ingredient in ingredient_ids:
            document_frequency = len(
                ingredient_to_drugs[ingredient]
            )
            ingredient_idf.append(
                math.log(
                    (1.0 + num_ingredient_drugs)
                    / (1.0 + document_frequency)
                )
                + 1.0
            )

        drug_rows = []
        drug_columns = []
        drug_values = []

        for drug, ingredients in sorted(
            drug_to_ingredients.items()
        ):
            for ingredient in sorted(ingredients):
                local_id = ingredient_to_local[
                    ingredient
                ]
                drug_rows.append(drug)
                drug_columns.append(local_id)
                drug_values.append(
                    ingredient_idf[local_id]
                )

        adr_ingredient_counts = Counter()

        for adr, supporting_drugs in adr_to_drugs.items():
            for drug in supporting_drugs:
                for ingredient in drug_to_ingredients.get(
                    drug, ()
                ):
                    local_id = ingredient_to_local[
                        ingredient
                    ]
                    adr_ingredient_counts[
                        (adr, local_id)
                    ] += 1

        adr_rows = []
        adr_columns = []
        adr_values = []

        for (adr, local_id), count in sorted(
            adr_ingredient_counts.items()
        ):
            adr_rows.append(adr)
            adr_columns.append(local_id)
            adr_values.append(
                float(count)
                * ingredient_idf[local_id]
            )

        def sparse_matrix(
            rows,
            columns,
            values,
        ):
            if rows:
                indices = torch.tensor(
                    [rows, columns],
                    dtype=torch.long,
                )
                sparse_values = torch.tensor(
                    values,
                    dtype=torch.float,
                )
            else:
                indices = torch.empty(
                    (2, 0),
                    dtype=torch.long,
                )
                sparse_values = torch.empty(
                    (0,),
                    dtype=torch.float,
                )

            return torch.sparse_coo_tensor(
                indices,
                sparse_values,
                size=(
                    self.num_node,
                    len(ingredient_ids),
                ),
            ).coalesce()

        drug_ingredient = sparse_matrix(
            drug_rows,
            drug_columns,
            drug_values,
        )
        adr_ingredient = sparse_matrix(
            adr_rows,
            adr_columns,
            adr_values,
        )

        raw_drug_counts = torch.zeros(
            self.num_node,
            dtype=torch.float,
        )
        raw_adr_occurrences = torch.zeros(
            self.num_node,
            dtype=torch.float,
        )

        for drug, ingredients in drug_to_ingredients.items():
            raw_drug_counts[drug] = len(ingredients)

        for adr, supporting_drugs in adr_to_drugs.items():
            raw_adr_occurrences[adr] = sum(
                len(drug_to_ingredients.get(drug, ()))
                for drug in supporting_drugs
            )

        known_cause_hashes = sorted(
            drug * self.num_node + adr
            for drug, adrs in drug_to_adrs.items()
            for adr in adrs
        )

        exact_path_counts = Counter()

        for ingredient, sharing_drugs in (
            ingredient_to_drugs.items()
        ):
            for query_drug in sharing_drugs:
                for supporting_drug in sharing_drugs:
                    if query_drug == supporting_drug:
                        continue

                    for adr in drug_to_adrs.get(
                        supporting_drug, ()
                    ):
                        pair_hash = (
                            query_drug * self.num_node
                            + adr
                        )
                        exact_path_counts[pair_hash] += 1

        exact_path_hashes = sorted(exact_path_counts)
        exact_path_values = [
            float(exact_path_counts[pair_hash])
            for pair_hash in exact_path_hashes
        ]

        self.ingredient_embedding = nn.Embedding(
            len(ingredient_ids),
            self.slot_dim,
        )
        self.slot_assignment_logits = nn.Parameter(
            torch.zeros(
                len(ingredient_ids),
                self.num_slots,
            )
        )
        self.query_projection = nn.Linear(
            self.slot_dim,
            self.slot_dim,
            bias=False,
        )
        self.candidate_projection = nn.Linear(
            self.slot_dim,
            self.slot_dim,
            bias=False,
        )

        nn.init.normal_(
            self.ingredient_embedding.weight,
            mean=0.0,
            std=1.0 / math.sqrt(self.slot_dim),
        )
        nn.init.normal_(
            self.slot_assignment_logits,
            mean=0.0,
            std=0.02,
        )
        nn.init.xavier_uniform_(
            self.query_projection.weight
        )
        nn.init.xavier_uniform_(
            self.candidate_projection.weight
        )

        self.register_buffer(
            "ingredient_entity_ids",
            torch.tensor(
                ingredient_ids,
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "ingredient_idf",
            torch.tensor(
                ingredient_idf,
                dtype=torch.float,
            ),
        )
        self.register_buffer(
            "drug_ingredient",
            drug_ingredient,
        )
        self.register_buffer(
            "adr_ingredient",
            adr_ingredient,
        )
        self.register_buffer(
            "raw_drug_ingredient_counts",
            raw_drug_counts,
        )
        self.register_buffer(
            "raw_adr_ingredient_occurrences",
            raw_adr_occurrences,
        )
        self.register_buffer(
            "known_cause_hashes",
            torch.tensor(
                known_cause_hashes,
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "exact_path_hashes",
            torch.tensor(
                exact_path_hashes,
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "exact_path_values",
            torch.tensor(
                exact_path_values,
                dtype=torch.float,
            ),
        )

    @staticmethod
    def _membership(sorted_hashes, query_hashes):
        if sorted_hashes.numel() == 0:
            return torch.zeros_like(
                query_hashes,
                dtype=torch.bool,
            )

        positions = torch.searchsorted(
            sorted_hashes,
            query_hashes,
        )
        valid = positions < sorted_hashes.numel()
        safe_positions = positions.clamp(
            max=sorted_hashes.numel() - 1
        )

        return valid & (
            sorted_hashes[safe_positions]
            == query_hashes
        )

    def _lookup_exact_paths(self, pair_hashes):
        if self.exact_path_hashes.numel() == 0:
            return torch.zeros_like(
                pair_hashes,
                dtype=torch.float,
            )

        positions = torch.searchsorted(
            self.exact_path_hashes,
            pair_hashes,
        )
        valid = (
            positions < self.exact_path_hashes.numel()
        )
        safe_positions = positions.clamp(
            max=self.exact_path_hashes.numel() - 1
        )
        matched = valid & (
            self.exact_path_hashes[safe_positions]
            == pair_hashes
        )

        values = torch.zeros_like(
            pair_hashes,
            dtype=self.exact_path_values.dtype,
        )
        values[matched] = self.exact_path_values[
            safe_positions[matched]
        ]
        return values

    def forward(
        self,
        batch,
        return_diagnostics=False,
    ):
        if batch.ndim < 2 or batch.shape[-1] != 3:
            raise ValueError(
                "batch must have shape "
                "[..., num_candidates, 3]"
            )

        h_index, t_index, r_index = batch.unbind(-1)
        original_shape = h_index.shape

        flat_h = h_index.reshape(-1)
        flat_t = t_index.reshape(-1)
        flat_r = r_index.reshape(-1)

        assignment = F.softmax(
            self.slot_assignment_logits
            / self.slot_temperature,
            dim=-1,
        )

        ingredient_slot_features = (
            self.ingredient_embedding.weight[
                :, None, :
            ]
            * assignment[:, :, None]
        )
        flattened_slot_features = (
            ingredient_slot_features.reshape(
                ingredient_slot_features.shape[0],
                self.num_slots * self.slot_dim,
            )
        )

        drug_slot_sums = torch.sparse.mm(
            self.drug_ingredient,
            flattened_slot_features,
        ).view(
            self.num_node,
            self.num_slots,
            self.slot_dim,
        )
        adr_slot_sums = torch.sparse.mm(
            self.adr_ingredient,
            flattened_slot_features,
        ).view(
            self.num_node,
            self.num_slots,
            self.slot_dim,
        )

        drug_slot_mass = torch.sparse.mm(
            self.drug_ingredient,
            assignment,
        )
        adr_slot_mass = torch.sparse.mm(
            self.adr_ingredient,
            assignment,
        )

        query_sums = drug_slot_sums.index_select(
            0,
            flat_h,
        )
        query_mass = drug_slot_mass.index_select(
            0,
            flat_h,
        )
        candidate_sums = adr_slot_sums.index_select(
            0,
            flat_t,
        )
        candidate_mass = adr_slot_mass.index_select(
            0,
            flat_t,
        )

        pair_hashes = (
            flat_h * self.num_node + flat_t
        )
        known_cause = self._membership(
            self.known_cause_hashes,
            pair_hashes,
        )
        known_float = known_cause.to(
            candidate_sums.dtype
        )

        loo_candidate_sums = (
            candidate_sums
            - known_float[:, None, None]
            * query_sums
        )
        loo_candidate_mass = (
            candidate_mass
            - known_float[:, None] * query_mass
        ).clamp_min(0.0)

        query_means = (
            query_sums
            / query_mass.clamp_min(1e-12)[
                :, :, None
            ]
        )
        candidate_means = (
            loo_candidate_sums
            / loo_candidate_mass.clamp_min(1e-12)[
                :, :, None
            ]
        )

        query_vectors = F.normalize(
            self.query_projection(query_means),
            dim=-1,
        )
        candidate_vectors = F.normalize(
            self.candidate_projection(
                candidate_means
            ),
            dim=-1,
        )

        slot_similarity = (
            query_vectors * candidate_vectors
        ).sum(dim=-1)

        valid_slots = (
            (query_mass > 0)
            & (loo_candidate_mass > 0)
        )
        attention_logits = (
            slot_similarity / self.slot_temperature
        ).masked_fill(
            ~valid_slots,
            -1e9,
        )
        slot_attention = F.softmax(
            attention_logits,
            dim=-1,
        )
        soft_score = (
            slot_attention
            * slot_similarity
            * valid_slots.to(slot_similarity.dtype)
        ).sum(dim=-1)

        exact_path_count = self._lookup_exact_paths(
            pair_hashes
        )
        exact_score = torch.log1p(
            exact_path_count
        )

        target_mask = (
            flat_r == self.target_relation_id
        )
        target_float = target_mask.to(
            soft_score.dtype
        )

        soft_score = soft_score * target_float
        exact_score = exact_score * target_float

        query_count = (
            self.raw_drug_ingredient_counts
            .index_select(0, flat_h)
        )
        candidate_occurrences = (
            self.raw_adr_ingredient_occurrences
            .index_select(0, flat_t)
        )
        loo_candidate_occurrences = (
            candidate_occurrences
            - known_float * query_count
        ).clamp_min(0.0)

        has_query = query_mass.sum(
            dim=-1
        ) > 0
        has_candidate = loo_candidate_mass.sum(
            dim=-1
        ) > 0

        evidence_logits = torch.stack(
            [
                exact_score,
                soft_score,
            ],
            dim=-1,
        ).view(
            *original_shape,
            2,
        )

        router_features = torch.stack(
            [
                exact_score,
                torch.log1p(query_count),
                torch.log1p(
                    loo_candidate_occurrences
                ),
                soft_score,
                has_query.to(soft_score.dtype),
                has_candidate.to(soft_score.dtype),
            ],
            dim=-1,
        ).view(
            *original_shape,
            6,
        )

        if not return_diagnostics:
            return evidence_logits, router_features

        diagnostics = {
            "known_cause":
                known_cause.view(original_shape),
            "exact_path_count":
                exact_path_count.view(original_shape),
            "slot_similarity":
                slot_similarity.view(
                    *original_shape,
                    self.num_slots,
                ),
            "slot_attention":
                slot_attention.view(
                    *original_shape,
                    self.num_slots,
                ),
            "query_slot_mass":
                query_mass.view(
                    *original_shape,
                    self.num_slots,
                ),
            "loo_candidate_slot_mass":
                loo_candidate_mass.view(
                    *original_shape,
                    self.num_slots,
                ),
            "soft_score":
                soft_score.view(original_shape),
        }

        return (
            evidence_logits,
            router_features,
            diagnostics,
        )

    def identity(self):
        return {
            "ingredient_expert_mode":
                "idf_multislot_leave_one_out",
            "num_slots": self.num_slots,
            "slot_dim": self.slot_dim,
            "slot_temperature":
                self.slot_temperature,
            "num_ingredients": int(
                self.ingredient_entity_ids.numel()
            ),
            "known_cause_pairs": int(
                self.known_cause_hashes.numel()
            ),
            "exact_evidence_pairs": int(
                self.exact_path_hashes.numel()
            ),
            "leave_one_drug_out": True,
            "training_graph_only": True,
        }



class OrganEvidenceExpert(nn.Module):
    """
    Training-graph-only organ evidence expert.

    The query drug organ profile is induced from its known training ADRs.
    For a known training cause pair, the candidate ADR contribution is
    removed from the query profile before overlap is computed.
    """

    def __init__(
        self,
        train_data,
        num_relation,
        affects_relation_id=1,
        target_relation_id=2,
    ):
        super().__init__()

        from collections import Counter, defaultdict

        if num_relation < 2:
            raise ValueError(
                "num_relation must include inverse relations"
            )

        self.num_node = int(train_data.num_nodes)
        self.num_relation = int(num_relation)
        self.affects_relation_id = int(
            affects_relation_id
        )
        self.target_relation_id = int(
            target_relation_id
        )

        edge_index = train_data.edge_index.detach().cpu()
        edge_type = train_data.edge_type.detach().cpu()

        affects_edges = edge_index[
            :, edge_type == self.affects_relation_id
        ].t().tolist()
        cause_edges = edge_index[
            :, edge_type == self.target_relation_id
        ].t().tolist()

        adr_to_organs = defaultdict(set)
        drug_to_adrs = defaultdict(set)

        for adr, organ in affects_edges:
            adr_to_organs[int(adr)].add(int(organ))

        for drug, adr in cause_edges:
            drug_to_adrs[int(drug)].add(int(adr))

        organ_ids = sorted({
            organ
            for organs in adr_to_organs.values()
            for organ in organs
        })
        organ_to_local = {
            organ: local_id
            for local_id, organ in enumerate(
                organ_ids
            )
        }

        num_organ_adrs = max(
            1,
            len(adr_to_organs),
        )
        organ_document_frequency = Counter()

        for organs in adr_to_organs.values():
            for organ in organs:
                organ_document_frequency[organ] += 1

        organ_idf = [
            (
                math.log(
                    (1.0 + num_organ_adrs)
                    / (
                        1.0
                        + organ_document_frequency[
                            organ
                        ]
                    )
                )
                + 1.0
            )
            for organ in organ_ids
        ]

        adr_organ = torch.zeros(
            self.num_node,
            len(organ_ids),
            dtype=torch.float,
        )

        for adr, organs in adr_to_organs.items():
            for organ in organs:
                local_id = organ_to_local[organ]
                adr_organ[adr, local_id] = (
                    organ_idf[local_id]
                )

        drug_organ = torch.zeros_like(
            adr_organ
        )

        for drug, adrs in drug_to_adrs.items():
            for adr in adrs:
                drug_organ[drug] += adr_organ[adr]

        known_cause_hashes = sorted(
            drug * self.num_node + adr
            for drug, adrs in drug_to_adrs.items()
            for adr in adrs
        )

        self.register_buffer(
            "organ_entity_ids",
            torch.tensor(
                organ_ids,
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "organ_idf",
            torch.tensor(
                organ_idf,
                dtype=torch.float,
            ),
        )
        self.register_buffer(
            "adr_organ",
            adr_organ,
        )
        self.register_buffer(
            "drug_organ",
            drug_organ,
        )
        self.register_buffer(
            "known_cause_hashes",
            torch.tensor(
                known_cause_hashes,
                dtype=torch.long,
            ),
        )

    @staticmethod
    def _membership(
        sorted_hashes,
        query_hashes,
    ):
        if sorted_hashes.numel() == 0:
            return torch.zeros_like(
                query_hashes,
                dtype=torch.bool,
            )

        positions = torch.searchsorted(
            sorted_hashes,
            query_hashes,
        )
        valid = positions < sorted_hashes.numel()
        safe_positions = positions.clamp(
            max=sorted_hashes.numel() - 1
        )

        return valid & (
            sorted_hashes[safe_positions]
            == query_hashes
        )

    def forward(
        self,
        batch,
        return_diagnostics=False,
    ):
        if batch.ndim < 2 or batch.shape[-1] != 3:
            raise ValueError(
                "batch must have shape "
                "[..., num_candidates, 3]"
            )

        h_index, t_index, r_index = batch.unbind(-1)
        original_shape = h_index.shape

        flat_h = h_index.reshape(-1)
        flat_t = t_index.reshape(-1)
        flat_r = r_index.reshape(-1)

        query_profile = self.drug_organ.index_select(
            0,
            flat_h,
        )
        candidate_profile = (
            self.adr_organ.index_select(
                0,
                flat_t,
            )
        )

        pair_hashes = (
            flat_h * self.num_node + flat_t
        )
        known_cause = self._membership(
            self.known_cause_hashes,
            pair_hashes,
        )
        known_float = known_cause.to(
            query_profile.dtype
        )

        loo_query_profile = (
            query_profile
            - known_float[:, None]
            * candidate_profile
        ).clamp_min(0.0)

        query_present = loo_query_profile > 0
        candidate_present = candidate_profile > 0

        if self.organ_idf.numel() == 0:
            overlap_weight = torch.zeros(
                flat_h.shape[0],
                dtype=torch.float,
                device=flat_h.device,
            )
            union_weight = torch.zeros_like(
                overlap_weight
            )
            query_organ_count = torch.zeros_like(
                overlap_weight
            )
            candidate_organ_count = (
                torch.zeros_like(
                    overlap_weight
                )
            )
        else:
            idf = self.organ_idf[None, :]

            overlap_weight = (
                (
                    query_present
                    & candidate_present
                ).to(idf.dtype)
                * idf
            ).sum(dim=-1)

            union_weight = (
                (
                    query_present
                    | candidate_present
                ).to(idf.dtype)
                * idf
            ).sum(dim=-1)

            query_organ_count = (
                query_present.sum(
                    dim=-1
                ).to(torch.float)
            )
            candidate_organ_count = (
                candidate_present.sum(
                    dim=-1
                ).to(torch.float)
            )

        weighted_jaccard = (
            overlap_weight
            / union_weight.clamp_min(1e-12)
        )
        organ_score = (
            torch.log1p(overlap_weight)
            + weighted_jaccard
        )

        target_mask = (
            flat_r == self.target_relation_id
        )
        target_float = target_mask.to(
            organ_score.dtype
        )
        organ_score = organ_score * target_float

        router_features = torch.stack(
            [
                organ_score,
                torch.log1p(
                    query_organ_count
                ),
                torch.log1p(
                    candidate_organ_count
                ),
            ],
            dim=-1,
        ).view(
            *original_shape,
            3,
        )
        organ_score = organ_score.view(
            original_shape
        )

        if not return_diagnostics:
            return organ_score, router_features

        diagnostics = {
            "known_cause":
                known_cause.view(original_shape),
            "overlap_weight":
                overlap_weight.view(
                    original_shape
                ),
            "weighted_jaccard":
                weighted_jaccard.view(
                    original_shape
                ),
            "query_organ_count":
                query_organ_count.view(
                    original_shape
                ),
            "candidate_organ_count":
                candidate_organ_count.view(
                    original_shape
                ),
        }

        return (
            organ_score,
            router_features,
            diagnostics,
        )

    def identity(self):
        return {
            "organ_expert_mode":
                "idf_overlap_leave_one_out",
            "num_organs": int(
                self.organ_entity_ids.numel()
            ),
            "known_cause_pairs": int(
                self.known_cause_hashes.numel()
            ),
            "affects_relation_id":
                self.affects_relation_id,
            "target_relation_id":
                self.target_relation_id,
            "leave_one_adr_out": True,
            "training_graph_only": True,
            "trainable_parameter_count": 0,
        }

def hybrid_candidate_list_loss(
    prediction,
    batch,
    target_relation_id,
    adversarial_temperature=1.0,
    positive_index=0,
    return_diagnostics=False,
):
    """
    Hybrid training loss for NBFNet-EviMoE.

    Target-relation rows contain log probabilities over the candidate
    list and use listwise negative log likelihood. Other relation rows
    retain the original adversarially weighted binary cross entropy.
    """
    if prediction.ndim != 2:
        raise ValueError(
            "prediction must have shape "
            "[batch_size, num_candidates]"
        )
    if batch.ndim != 3 or batch.shape[-1] != 3:
        raise ValueError(
            "batch must have shape "
            "[batch_size, num_candidates, 3]"
        )
    if tuple(batch.shape[:2]) != tuple(
        prediction.shape
    ):
        raise ValueError(
            "batch candidate dimensions must match "
            "prediction"
        )

    batch_size, num_candidates = prediction.shape

    if batch_size < 1 or num_candidates < 2:
        raise ValueError(
            "hybrid loss requires at least one row "
            "and two candidates"
        )
    if not 0 <= positive_index < num_candidates:
        raise ValueError(
            "positive_index is outside the "
            "candidate list"
        )
    if adversarial_temperature < 0:
        raise ValueError(
            "adversarial_temperature must be "
            "nonnegative"
        )

    relation_index = batch[..., 2]
    row_relation = relation_index[:, 0]

    if not torch.equal(
        relation_index,
        row_relation[:, None].expand_as(
            relation_index
        ),
    ):
        raise ValueError(
            "all candidates in one row must use "
            "the same relation"
        )

    target_rows = (
        row_relation == int(target_relation_id)
    )
    non_target_rows = ~target_rows

    row_losses = prediction.new_zeros(
        batch_size
    )

    if target_rows.any():
        target_prediction = prediction[
            target_rows
        ]
        row_losses[target_rows] = (
            -target_prediction[:, positive_index]
        )

    if non_target_rows.any():
        non_target_prediction = prediction[
            non_target_rows
        ]

        binary_target = torch.zeros_like(
            non_target_prediction
        )
        binary_target[:, positive_index] = 1

        element_loss = (
            F.binary_cross_entropy_with_logits(
                non_target_prediction,
                binary_target,
                reduction="none",
            )
        )

        negative_weight = torch.ones_like(
            non_target_prediction
        )

        if adversarial_temperature > 0:
            with torch.no_grad():
                negative_weight[:, 1:] = F.softmax(
                    non_target_prediction[:, 1:]
                    / adversarial_temperature,
                    dim=-1,
                )
        else:
            negative_weight[:, 1:] = (
                1.0 / (num_candidates - 1)
            )

        row_losses[non_target_rows] = (
            (
                element_loss * negative_weight
            ).sum(dim=-1)
            / negative_weight.sum(dim=-1)
        )

    loss = row_losses.mean()

    if not return_diagnostics:
        return loss

    diagnostics = {
        "target_row_count": int(
            target_rows.sum().item()
        ),
        "non_target_row_count": int(
            non_target_rows.sum().item()
        ),
        "target_loss_mean": (
            float(
                row_losses[target_rows]
                .detach().mean().cpu()
            )
            if target_rows.any()
            else None
        ),
        "non_target_loss_mean": (
            float(
                row_losses[non_target_rows]
                .detach().mean().cpu()
            )
            if non_target_rows.any()
            else None
        ),
    }
    return loss, diagnostics


def hybrid_pairwise_rerank_loss(
    prediction,
    batch,
    target_relation_id,
    adversarial_temperature=1.0,
    positive_index=0,
    return_diagnostics=False,
):
    """
    Hybrid loss for set-invariant evidence reranking.

    Target-relation rows use pairwise logistic ranking over raw logits.
    Other relation rows retain the original adversarially weighted BCE
    used by NBFNet.
    """
    if prediction.ndim != 2:
        raise ValueError(
            "prediction must have shape "
            "[batch_size, num_candidates]"
        )
    if batch.ndim != 3 or batch.shape[-1] != 3:
        raise ValueError(
            "batch must have shape "
            "[batch_size, num_candidates, 3]"
        )
    if tuple(batch.shape[:2]) != tuple(
        prediction.shape
    ):
        raise ValueError(
            "batch candidate dimensions must match "
            "prediction"
        )

    batch_size, num_candidates = prediction.shape

    if batch_size < 1 or num_candidates < 2:
        raise ValueError(
            "hybrid loss requires at least one row "
            "and two candidates"
        )
    if not 0 <= positive_index < num_candidates:
        raise ValueError(
            "positive_index is outside the "
            "candidate list"
        )
    if adversarial_temperature < 0:
        raise ValueError(
            "adversarial_temperature must be "
            "nonnegative"
        )

    relation_index = batch[..., 2]
    row_relation = relation_index[:, 0]

    if not torch.equal(
        relation_index,
        row_relation[:, None].expand_as(
            relation_index
        ),
    ):
        raise ValueError(
            "all candidates in one row must use "
            "the same relation"
        )

    target_rows = (
        row_relation == int(target_relation_id)
    )
    non_target_rows = ~target_rows
    row_losses = prediction.new_zeros(batch_size)

    target_pair_count = 0

    if target_rows.any():
        target_prediction = prediction[target_rows]
        positive = target_prediction[
            :, positive_index
        ].unsqueeze(1)

        negative_mask = torch.ones(
            num_candidates,
            dtype=torch.bool,
            device=prediction.device,
        )
        negative_mask[positive_index] = False
        negatives = target_prediction[
            :, negative_mask
        ]

        pairwise_margin = positive - negatives
        pairwise_loss = F.softplus(
            -pairwise_margin
        )
        row_losses[target_rows] = (
            pairwise_loss.mean(dim=-1)
        )
        target_pair_count = int(
            pairwise_loss.numel()
        )

    if non_target_rows.any():
        non_target_prediction = prediction[
            non_target_rows
        ]

        binary_target = torch.zeros_like(
            non_target_prediction
        )
        binary_target[:, positive_index] = 1

        element_loss = (
            F.binary_cross_entropy_with_logits(
                non_target_prediction,
                binary_target,
                reduction="none",
            )
        )

        negative_weight = torch.ones_like(
            non_target_prediction
        )

        if adversarial_temperature > 0:
            with torch.no_grad():
                negative_weight[:, 1:] = F.softmax(
                    non_target_prediction[:, 1:]
                    / adversarial_temperature,
                    dim=-1,
                )
        else:
            negative_weight[:, 1:] = (
                1.0 / (num_candidates - 1)
            )

        row_losses[non_target_rows] = (
            (
                element_loss * negative_weight
            ).sum(dim=-1)
            / negative_weight.sum(dim=-1)
        )

    loss = row_losses.mean()

    if not return_diagnostics:
        return loss

    diagnostics = {
        "loss_mode":
            "hybrid_pairwise_reranking",
        "target_row_count": int(
            target_rows.sum().item()
        ),
        "non_target_row_count": int(
            non_target_rows.sum().item()
        ),
        "target_pair_count":
            target_pair_count,
        "target_loss_mean": (
            float(
                row_losses[target_rows]
                .detach().mean().cpu()
            )
            if target_rows.any()
            else None
        ),
        "non_target_loss_mean": (
            float(
                row_losses[non_target_rows]
                .detach().mean().cpu()
            )
            if non_target_rows.any()
            else None
        ),
    }
    return loss, diagnostics


def candidate_list_nll(
    mixture_log_probabilities,
    positive_index=0,
):
    if mixture_log_probabilities.ndim != 2:
        raise ValueError(
            "mixture_log_probabilities must have shape "
            "[batch_size, num_candidates]"
        )

    num_candidates = mixture_log_probabilities.shape[1]
    if not 0 <= positive_index < num_candidates:
        raise ValueError(
            "positive_index is outside the candidate list"
        )

    return -mixture_log_probabilities[
        :, positive_index
    ].mean()


class SetInvariantEvidenceReranker(nn.Module):
    """
    Candidate-independent evidence reranker.

    Unlike candidate probability mixtures, each candidate correction is
    computed independently and therefore does not change when unrelated
    candidates are added to or removed from the candidate list.

    The correction head is initialized to zero, so the initial model
    exactly preserves the pretrained NBFNet ranking.
    """

    def __init__(
        self,
        num_evidence_experts,
        router_feature_dim,
        hidden_dim=32,
        maximum_correction=1.0,
    ):
        super().__init__()

        if num_evidence_experts < 1:
            raise ValueError(
                "num_evidence_experts must be at least 1"
            )
        if router_feature_dim < 1:
            raise ValueError(
                "router_feature_dim must be at least 1"
            )
        if hidden_dim < 1:
            raise ValueError(
                "hidden_dim must be at least 1"
            )
        if maximum_correction <= 0:
            raise ValueError(
                "maximum_correction must be greater than 0"
            )

        self.num_evidence_experts = int(
            num_evidence_experts
        )
        self.router_feature_dim = int(
            router_feature_dim
        )
        self.hidden_dim = int(hidden_dim)
        self.maximum_correction = float(
            maximum_correction
        )
        self.input_dim = (
            self.num_evidence_experts
            + self.router_feature_dim
        )

        self.feature_norm = nn.LayerNorm(
            self.input_dim
        )
        self.hidden = nn.Linear(
            self.input_dim,
            self.hidden_dim,
        )
        self.output = nn.Linear(
            self.hidden_dim,
            2,
        )

        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        base_logits,
        evidence_logits,
        router_features,
        return_diagnostics=False,
    ):
        if base_logits.ndim != 2:
            raise ValueError(
                "base_logits must have shape "
                "[batch_size, num_candidates]"
            )

        expected_evidence_shape = (
            *base_logits.shape,
            self.num_evidence_experts,
        )
        if tuple(evidence_logits.shape) != (
            expected_evidence_shape
        ):
            raise ValueError(
                "evidence_logits must have shape %r, got %r"
                % (
                    expected_evidence_shape,
                    tuple(evidence_logits.shape),
                )
            )

        expected_router_shape = (
            *base_logits.shape,
            self.router_feature_dim,
        )
        if tuple(router_features.shape) != (
            expected_router_shape
        ):
            raise ValueError(
                "router_features must have shape %r, got %r"
                % (
                    expected_router_shape,
                    tuple(router_features.shape),
                )
            )

        candidate_features = torch.cat(
            [
                evidence_logits,
                router_features,
            ],
            dim=-1,
        )
        normalized_features = self.feature_norm(
            candidate_features
        )
        hidden = F.gelu(
            self.hidden(normalized_features)
        )
        raw_output = self.output(hidden)

        # tanh may saturate to exactly 1 in finite precision.
        # Use the nearest representable value below the configured
        # maximum to guarantee a strictly bounded residual.
        strict_maximum = torch.nextafter(
            raw_output.new_tensor(self.maximum_correction),
            raw_output.new_tensor(0.0),
        )
        correction = (
            strict_maximum
            * torch.tanh(raw_output[..., 0])
        )
        confidence = torch.sigmoid(
            raw_output[..., 1]
        )
        bounded_residual = (
            confidence * correction
        )
        reranked_logits = (
            base_logits + bounded_residual
        )

        if not return_diagnostics:
            return reranked_logits

        return reranked_logits, {
            "correction": correction,
            "confidence": confidence,
            "bounded_residual": bounded_residual,
        }

    def identity(self):
        return {
            "fusion_mode":
                "set_invariant_bounded_reranking",
            "num_evidence_experts":
                self.num_evidence_experts,
            "router_feature_dim":
                self.router_feature_dim,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "maximum_correction":
                self.maximum_correction,
            "zero_initialization": True,
            "candidate_set_invariant": True,
        }


def candidate_pairwise_logistic_loss(
    prediction,
    positive_index=0,
):
    """
    Pairwise logistic ranking loss for one positive and multiple negatives.

    The loss depends only on positive-negative score differences and is
    invariant to a common score shift and to negative-candidate order.
    """
    if prediction.ndim != 2:
        raise ValueError(
            "prediction must have shape "
            "[batch_size, num_candidates]"
        )
    if prediction.shape[1] < 2:
        raise ValueError(
            "prediction must contain at least "
            "one positive and one negative"
        )
    if not 0 <= positive_index < prediction.shape[1]:
        raise ValueError(
            "positive_index is outside the candidate list"
        )

    positive = prediction[
        :, positive_index
    ].unsqueeze(1)

    negative_mask = torch.ones(
        prediction.shape[1],
        dtype=torch.bool,
        device=prediction.device,
    )
    negative_mask[positive_index] = False
    negatives = prediction[:, negative_mask]

    pairwise_margin = positive - negatives
    return F.softplus(
        -pairwise_margin
    ).mean()
