import math
from collections import Counter, defaultdict

import torch
from torch import nn
from torch.nn import functional as F


def _sparse_matrix(rows, columns, values, shape):
    if rows:
        indices = torch.tensor(
            [rows, columns],
            dtype=torch.long,
        )
        values = torch.tensor(values, dtype=torch.float)
    else:
        indices = torch.empty((2, 0), dtype=torch.long)
        values = torch.empty((0,), dtype=torch.float)

    return torch.sparse_coo_tensor(
        indices,
        values,
        size=shape,
    ).coalesce()


class SparseIngredientPrototypeRefiner(nn.Module):
    """
    Candidate-level residual refinement for sparse ingredient evidence.

    ADR prototypes are constructed only from the training fact graph.
    For a known training cause pair, the query drug contribution is
    subtracted from the candidate prototype (leave-one-drug-out).
    Refinement is active only when no other drug provides an exact
    shared-ingredient path.
    """

    def __init__(
        self,
        train_data,
        num_relation,
        prototype_dim=8,
        prototype_scale=0.1,
        prototype_scale_max=1.0,
        contains_relation_id=0,
        target_relation_id=2,
    ):
        super().__init__()

        if prototype_dim < 1:
            raise ValueError("prototype_dim must be at least 1")
        if prototype_scale <= 0:
            raise ValueError("prototype_scale must be greater than 0")
        if prototype_scale_max <= 0:
            raise ValueError(
                "prototype_scale_max must be greater than 0"
            )
        if prototype_scale >= prototype_scale_max:
            raise ValueError(
                "prototype_scale must be smaller than "
                "prototype_scale_max"
            )
        if num_relation < 2:
            raise ValueError("num_relation must include inverse relations")

        self.num_node = int(train_data.num_nodes)
        self.num_relation = int(num_relation)
        self.prototype_dim = int(prototype_dim)
        self.prototype_scale_initial = float(
            prototype_scale
        )
        self.prototype_scale_max = float(
            prototype_scale_max
        )
        self.contains_relation_id = int(contains_relation_id)
        self.target_relation_id = int(target_relation_id)

        edge_index = train_data.edge_index.detach().cpu()
        edge_type = train_data.edge_type.detach().cpu()

        contains_mask = edge_type == self.contains_relation_id
        cause_mask = edge_type == self.target_relation_id

        contains_edges = edge_index[:, contains_mask].t().tolist()
        cause_edges = edge_index[:, cause_mask].t().tolist()

        drug_to_ingredients = defaultdict(set)
        ingredient_to_drugs = defaultdict(set)
        drug_to_adrs = defaultdict(set)
        adr_to_drugs = defaultdict(set)

        for drug, ingredient in contains_edges:
            drug_to_ingredients[int(drug)].add(int(ingredient))
            ingredient_to_drugs[int(ingredient)].add(int(drug))

        for drug, adr in cause_edges:
            drug_to_adrs[int(drug)].add(int(adr))
            adr_to_drugs[int(adr)].add(int(drug))

        ingredient_ids = sorted(ingredient_to_drugs)
        if not ingredient_ids:
            raise ValueError("training graph contains no ingredient entities")

        ingredient_to_local = {
            entity_id: local_id
            for local_id, entity_id in enumerate(ingredient_ids)
        }

        drug_rows = []
        drug_columns = []

        for drug, ingredients in sorted(drug_to_ingredients.items()):
            for ingredient in sorted(ingredients):
                drug_rows.append(drug)
                drug_columns.append(
                    ingredient_to_local[ingredient]
                )

        drug_values = [1.0] * len(drug_rows)

        adr_ingredient_counts = Counter()

        for adr, supporting_drugs in adr_to_drugs.items():
            for drug in supporting_drugs:
                for ingredient in drug_to_ingredients.get(drug, ()):
                    adr_ingredient_counts[
                        (adr, ingredient_to_local[ingredient])
                    ] += 1

        adr_rows = []
        adr_columns = []
        adr_values = []

        for (adr, local_ingredient), count in sorted(
            adr_ingredient_counts.items()
        ):
            adr_rows.append(adr)
            adr_columns.append(local_ingredient)
            adr_values.append(float(count))

        drug_ingredient = _sparse_matrix(
            drug_rows,
            drug_columns,
            drug_values,
            (self.num_node, len(ingredient_ids)),
        )
        adr_ingredient = _sparse_matrix(
            adr_rows,
            adr_columns,
            adr_values,
            (self.num_node, len(ingredient_ids)),
        )

        drug_counts = torch.sparse.sum(
            drug_ingredient,
            dim=1,
        ).to_dense()
        adr_counts = torch.sparse.sum(
            adr_ingredient,
            dim=1,
        ).to_dense()

        known_cause_hashes = sorted(
            {
                drug * self.num_node + adr
                for drug, adrs in drug_to_adrs.items()
                for adr in adrs
            }
        )

        exact_evidence_hashes = set()

        for ingredient, sharing_drugs in ingredient_to_drugs.items():
            for supporting_drug in sharing_drugs:
                for adr in drug_to_adrs.get(supporting_drug, ()):
                    for query_drug in sharing_drugs:
                        if query_drug != supporting_drug:
                            exact_evidence_hashes.add(
                                query_drug * self.num_node + adr
                            )

        self.ingredient_embedding = nn.Embedding(
            len(ingredient_ids),
            self.prototype_dim,
        )
        self.query_projection = nn.Linear(
            self.prototype_dim,
            self.prototype_dim,
            bias=False,
        )
        self.prototype_projection = nn.Linear(
            self.prototype_dim,
            self.prototype_dim,
            bias=False,
        )

        nn.init.normal_(
            self.ingredient_embedding.weight,
            mean=0.0,
            std=1.0 / math.sqrt(self.prototype_dim),
        )
        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.xavier_uniform_(self.prototype_projection.weight)

        initial_fraction = (
            self.prototype_scale_initial
            / self.prototype_scale_max
        )
        initial_scale_logit = math.log(
            initial_fraction
            / (1.0 - initial_fraction)
        )
        self.scale_logit = nn.Parameter(
            torch.tensor(
                initial_scale_logit,
                dtype=torch.float,
            )
        )

        self.register_buffer(
            "ingredient_entity_ids",
            torch.tensor(ingredient_ids, dtype=torch.long),
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
            "drug_ingredient_counts",
            drug_counts,
        )
        self.register_buffer(
            "adr_ingredient_counts",
            adr_counts,
        )
        self.register_buffer(
            "known_cause_hashes",
            torch.tensor(
                known_cause_hashes,
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "exact_evidence_hashes",
            torch.tensor(
                sorted(exact_evidence_hashes),
                dtype=torch.long,
            ),
        )

    @staticmethod
    def _hash_membership(sorted_hashes, query_hashes):
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
            sorted_hashes[safe_positions] == query_hashes
        )

    def _components(self, batch):
        if batch.ndim < 2 or batch.shape[-1] != 3:
            raise ValueError(
                "batch must have shape [..., candidates, 3]"
            )

        h_index, t_index, r_index = batch.unbind(-1)
        original_shape = h_index.shape

        flat_h = h_index.reshape(-1)
        flat_t = t_index.reshape(-1)
        flat_r = r_index.reshape(-1)

        ingredient_weight = self.ingredient_embedding.weight

        drug_sums = torch.sparse.mm(
            self.drug_ingredient,
            ingredient_weight,
        )
        adr_sums = torch.sparse.mm(
            self.adr_ingredient,
            ingredient_weight,
        )

        query_sums = drug_sums.index_select(0, flat_h)
        query_counts = self.drug_ingredient_counts.index_select(
            0,
            flat_h,
        )

        prototype_sums = adr_sums.index_select(0, flat_t)
        prototype_counts = self.adr_ingredient_counts.index_select(
            0,
            flat_t,
        )

        pair_hashes = flat_h * self.num_node + flat_t

        known_cause = self._hash_membership(
            self.known_cause_hashes,
            pair_hashes,
        )
        exact_evidence = self._hash_membership(
            self.exact_evidence_hashes,
            pair_hashes,
        )

        known_float = known_cause.to(prototype_sums.dtype)

        loo_prototype_sums = (
            prototype_sums
            - known_float.unsqueeze(-1) * query_sums
        )
        loo_prototype_counts = (
            prototype_counts
            - known_float * query_counts
        )

        query_means = query_sums / query_counts.clamp_min(
            1.0
        ).unsqueeze(-1)
        prototype_means = (
            loo_prototype_sums
            / loo_prototype_counts.clamp_min(1.0).unsqueeze(-1)
        )

        query_vectors = F.normalize(
            self.query_projection(query_means),
            dim=-1,
        )
        prototype_vectors = F.normalize(
            self.prototype_projection(prototype_means),
            dim=-1,
        )

        similarity = (
            query_vectors * prototype_vectors
        ).sum(dim=-1)

        active = (
            (flat_r == self.target_relation_id)
            & ~exact_evidence
            & (query_counts > 0)
            & (loo_prototype_counts > 0)
        )

        scale = (
            self.prototype_scale_max
            * torch.sigmoid(self.scale_logit)
        )
        correction = (
            scale
            * similarity
            * active.to(similarity.dtype)
        )

        return {
            "correction": correction.view(original_shape),
            "active": active.view(original_shape),
            "known_cause": known_cause.view(original_shape),
            "exact_evidence": exact_evidence.view(original_shape),
            "query_count": query_counts.view(original_shape),
            "loo_prototype_count": loo_prototype_counts.view(
                original_shape
            ),
            "similarity": similarity.view(original_shape),
            "scale": scale,
        }

    def forward(self, batch, return_diagnostics=False):
        components = self._components(batch)

        if return_diagnostics:
            return components["correction"], components

        return components["correction"]

    def identity(self):
        return {
            "prototype_mode": "sparse_ingredient_bounded",
            "prototype_dim": self.prototype_dim,
            "prototype_scale_initial":
                self.prototype_scale_initial,
            "prototype_scale_max":
                self.prototype_scale_max,
            "prototype_scale_current": float(
                (
                    self.prototype_scale_max
                    * torch.sigmoid(self.scale_logit)
                ).detach().cpu()
            ),
            "num_ingredients": int(
                self.ingredient_entity_ids.numel()
            ),
            "known_cause_pairs": int(
                self.known_cause_hashes.numel()
            ),
            "exact_evidence_pairs": int(
                self.exact_evidence_hashes.numel()
            ),
            "leave_one_drug_out": True,
        }
