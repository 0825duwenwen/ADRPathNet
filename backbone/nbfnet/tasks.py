from functools import reduce

import torch


def edge_match(edge_index, query_index):
    # O((n + q)logn) time
    # O(n) memory
    # edge_index: big underlying graph
    # query_index: edges to match

    # preparing unique hashing of edges, base: (max_node, max_relation) + 1
    base = edge_index.max(dim=1)[0] + 1
    # we will map edges to long ints, so we need to make sure the maximum product is less than MAX_LONG_INT
    # idea: max number of edges = num_nodes * num_relations
    # e.g. for a graph of 10 nodes / 5 relations, edge IDs 0...9 mean all possible outgoing edge types from node 0
    # given a tuple (h, r), we will search for all other existing edges starting from head h
    assert reduce(int.__mul__, base.tolist()) < torch.iinfo(torch.long).max
    scale = base.cumprod(0)
    scale = scale[-1] // scale

    # hash both the original edge index and the query index to unique integers
    edge_hash = (edge_index * scale.unsqueeze(-1)).sum(dim=0)
    edge_hash, order = edge_hash.sort()
    query_hash = (query_index * scale.unsqueeze(-1)).sum(dim=0)

    # matched ranges: [start[i], end[i])
    start = torch.bucketize(query_hash, edge_hash)
    end = torch.bucketize(query_hash, edge_hash, right=True)
    # num_match shows how many edges satisfy the (h, r) pattern for each query in the batch
    num_match = end - start

    # generate the corresponding ranges
    offset = num_match.cumsum(0) - num_match
    range = torch.arange(num_match.sum(), device=edge_index.device)
    range = range + (start - offset).repeat_interleave(num_match)

    return order[range], num_match




# ADR-constrained negative sampling
def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _adr_negative_sampling_enabled(task_cfg):
    return bool(_cfg_get(task_cfg, "adr_negative_sampling", False))


def _load_adr_candidate_ids(path, device, num_nodes):
    cache_key = str(path)
    cache = getattr(_load_adr_candidate_ids, "_cache", {})
    if cache_key not in cache:
        adr_ids = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    adr_ids.append(int(line))
        cache[cache_key] = torch.tensor(adr_ids, dtype=torch.long)
        _load_adr_candidate_ids._cache = cache

    ids = cache[cache_key].to(device)
    ids = ids[(ids >= 0) & (ids < num_nodes)]
    if ids.numel() == 0:
        raise ValueError("No valid ADR candidate ids found for ADR negative sampling")
    return ids


def _load_target_relation_id(path, target_relation):
    import json

    cache_key = (str(path), str(target_relation))
    cache = getattr(_load_target_relation_id, "_cache", {})
    if cache_key not in cache:
        with open(path, "r", encoding="utf-8") as f:
            relation2id = json.load(f)
        if target_relation not in relation2id:
            raise KeyError(f"Target relation {target_relation!r} not found in {path}")
        cache[cache_key] = int(relation2id[target_relation])
        _load_target_relation_id._cache = cache
    return cache[cache_key]


def _known_true_tails(data, head_id, relation_id):
    mask = (data.edge_index[0] == head_id) & (data.edge_type == relation_id)
    return data.edge_index[1, mask]


def _sample_from_allowed_candidates(allowed_candidates, num_negative):
    if allowed_candidates.numel() == 0:
        raise ValueError("No allowed ADR negative tail candidates remain after filtering positives")
    rand_index = torch.randint(
        allowed_candidates.numel(),
        (num_negative,),
        device=allowed_candidates.device,
    )
    return allowed_candidates[rand_index]




def _load_protected_tails(path, device):
    import json

    cpu_cache_key = str(path)
    device_cache_key = (str(path), str(device))

    cpu_cache = getattr(_load_protected_tails, "_cpu_cache", {})
    if cpu_cache_key not in cpu_cache:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        protected_tails = {}
        for drug_id, adr_ids in raw.items():
            protected_tails[int(drug_id)] = torch.tensor(
                [int(adr_id) for adr_id in adr_ids],
                dtype=torch.long,
            )
        cpu_cache[cpu_cache_key] = protected_tails
        _load_protected_tails._cpu_cache = cpu_cache

    device_cache = getattr(_load_protected_tails, "_device_cache", {})
    if device_cache_key not in device_cache:
        device_cache[device_cache_key] = {
            drug_id: tails.to(device)
            for drug_id, tails in cpu_cache[cpu_cache_key].items()
        }
        _load_protected_tails._device_cache = device_cache

    return device_cache[device_cache_key]


def _build_evimoe_ingredient_hard_pools(
    data,
    target_relation_id,
    contains_relation_id=0,
):
    """
    Build query-drug hard ADR pools from the training graph only.

    A candidate ADR is hard for a query drug when another training drug
    shares at least one ingredient with the query and causes that ADR.
    The query drug itself is excluded as a supporting witness.
    """
    from collections import defaultdict

    cache_key = (
        int(data.edge_index.data_ptr()),
        int(target_relation_id),
        int(contains_relation_id),
        str(data.edge_index.device),
    )
    cache = getattr(
        _build_evimoe_ingredient_hard_pools,
        "_cache",
        {},
    )

    if cache_key in cache:
        return cache[cache_key]

    edge_index = data.edge_index.detach().cpu()
    edge_type = data.edge_type.detach().cpu()

    contains_edges = edge_index[
        :, edge_type == int(contains_relation_id)
    ].t().tolist()
    cause_edges = edge_index[
        :, edge_type == int(target_relation_id)
    ].t().tolist()

    ingredient_to_drugs = defaultdict(set)
    drug_to_adrs = defaultdict(set)

    for drug, ingredient in contains_edges:
        ingredient_to_drugs[
            int(ingredient)
        ].add(int(drug))

    for drug, adr in cause_edges:
        drug_to_adrs[int(drug)].add(int(adr))

    hard_sets = defaultdict(set)

    for sharing_drugs in ingredient_to_drugs.values():
        for query_drug in sharing_drugs:
            for supporting_drug in sharing_drugs:
                if query_drug == supporting_drug:
                    continue

                hard_sets[query_drug].update(
                    drug_to_adrs.get(
                        supporting_drug,
                        (),
                    )
                )

    device = data.edge_index.device
    hard_pools = {
        int(drug): torch.tensor(
            sorted(adrs),
            dtype=torch.long,
            device=device,
        )
        for drug, adrs in hard_sets.items()
        if adrs
    }

    cache[cache_key] = hard_pools
    _build_evimoe_ingredient_hard_pools._cache = (
        cache
    )
    return hard_pools



def _select_evimoe_hard_fraction(
    hard_pool,
    true_tail_id,
    exact_positive_fraction,
    sparse_positive_fraction,
):
    """
    Use fewer hard negatives for positives already supported by exact
    ingredient evidence and more for sparse positives.
    """
    for value, name in (
        (
            exact_positive_fraction,
            "exact_positive_fraction",
        ),
        (
            sparse_positive_fraction,
            "sparse_positive_fraction",
        ),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "%s must be between 0 and 1"
                % name
            )

    positive_has_exact_evidence = (
        hard_pool is not None
        and hard_pool.numel() > 0
        and bool(
            (
                hard_pool
                == int(true_tail_id)
            ).any()
        )
    )

    selected_fraction = (
        exact_positive_fraction
        if positive_has_exact_evidence
        else sparse_positive_fraction
    )

    return (
        float(selected_fraction),
        positive_has_exact_evidence,
    )

def _sample_evimoe_mixed_negatives(
    allowed_candidates,
    hard_candidates,
    num_negative,
    hard_fraction,
):
    if allowed_candidates.numel() == 0:
        raise ValueError(
            "no allowed candidates for EviMoE "
            "negative sampling"
        )
    if not 0.0 <= hard_fraction <= 1.0:
        raise ValueError(
            "EviMoE hard-negative fraction must "
            "be between 0 and 1"
        )

    requested_hard = int(
        round(num_negative * hard_fraction)
    )
    requested_hard = min(
        num_negative,
        max(0, requested_hard),
    )

    if (
        requested_hard == 0
        or hard_candidates is None
        or hard_candidates.numel() == 0
    ):
        return _sample_from_allowed_candidates(
            allowed_candidates,
            num_negative,
        )

    hard_samples = _sample_from_allowed_candidates(
        hard_candidates,
        requested_hard,
    )
    random_count = (
        num_negative - requested_hard
    )

    if random_count > 0:
        random_samples = (
            _sample_from_allowed_candidates(
                allowed_candidates,
                random_count,
            )
        )
        samples = torch.cat(
            [hard_samples, random_samples],
            dim=0,
        )
    else:
        samples = hard_samples

    permutation = torch.randperm(
        samples.numel(),
        device=samples.device,
    )
    return samples[permutation]


# ===== ACHNS v0.1.0 =====
# ADR Co-occurrence Hard Negative Sampling
#
# Hard pools are constructed only from the graph object passed
# by the training loop. In formal training this object is
# train_data. No validation/test targets are read here.

_ACHNS_POOL_CACHE = {}


def _build_achns_hard_pools(
    data,
    target_relation_id,
    adr_candidate_ids,
    pool_size=32,
):
    from collections import defaultdict, Counter

    pool_size = int(pool_size)

    if pool_size <= 0:
        raise ValueError(
            "ACHNS pool_size must be positive"
        )

    device = adr_candidate_ids.device

    cache_key = (
        id(data),
        int(target_relation_id),
        int(pool_size),
        int(data.num_nodes),
    )

    cached = _ACHNS_POOL_CACHE.get(cache_key)

    if cached is not None:
        return {
            key: value.to(device)
            for key, value in cached.items()
        }

    adr_cpu = set(
        int(x)
        for x in adr_candidate_ids.detach().cpu().tolist()
    )

    edge_index = data.edge_index.detach().cpu()
    edge_type = data.edge_type.detach().cpu()

    drug_to_adrs = defaultdict(set)
    adr_to_drugs = defaultdict(set)

    # IMPORTANT:
    # Use forward target relation only.
    # data contains train facts plus inverse edges, but inverse
    # relation IDs differ from target_relation_id.
    mask = edge_type == int(target_relation_id)

    heads = edge_index[0, mask].tolist()
    tails = edge_index[1, mask].tolist()

    for head_id, tail_id in zip(heads, tails):
        head_id = int(head_id)
        tail_id = int(tail_id)

        if tail_id not in adr_cpu:
            continue

        drug_to_adrs[head_id].add(tail_id)
        adr_to_drugs[tail_id].add(head_id)

    pair_intersection = Counter()

    for positive_adrs in drug_to_adrs.values():
        values = sorted(positive_adrs)

        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                pair_intersection[
                    (values[i], values[j])
                ] += 1

    neighbors = defaultdict(list)

    for (adr_a, adr_b), intersection in (
        pair_intersection.items()
    ):
        degree_a = len(adr_to_drugs[adr_a])
        degree_b = len(adr_to_drugs[adr_b])

        union = (
            degree_a
            + degree_b
            - intersection
        )

        if union <= 0:
            continue

        jaccard = float(intersection) / float(union)

        neighbors[adr_a].append(
            (
                adr_b,
                jaccard,
                intersection,
            )
        )

        neighbors[adr_b].append(
            (
                adr_a,
                jaccard,
                intersection,
            )
        )

    pools_cpu = {}

    for adr_id in adr_cpu:
        candidates = neighbors.get(adr_id, [])

        candidates.sort(
            key=lambda item: (
                -item[1],
                -item[2],
                item[0],
            )
        )

        top_ids = [
            int(item[0])
            for item in candidates[:pool_size]
        ]

        pools_cpu[int(adr_id)] = torch.tensor(
            top_ids,
            dtype=torch.long,
        )

    _ACHNS_POOL_CACHE[cache_key] = pools_cpu

    return {
        key: value.to(device)
        for key, value in pools_cpu.items()
    }


def _sample_achns_mixed_negatives(
    allowed_candidates,
    hard_candidates,
    num_negative,
    hard_count,
):
    """
    ACHNS v0.1.1

    Sample distinct hard negatives first, then distinct
    random negatives from the remaining strict ADR set.

    If fewer than hard_count hard candidates are available,
    the missing quota is filled with random ADR negatives.
    """

    num_negative = int(num_negative)
    hard_count = int(hard_count)

    if num_negative <= 0:
        raise ValueError(
            "num_negative must be positive"
        )

    if hard_count < 0 or hard_count > num_negative:
        raise ValueError(
            "ACHNS hard_count must satisfy "
            "0 <= hard_count <= num_negative"
        )

    if allowed_candidates.numel() == 0:
        raise ValueError(
            "ACHNS received no allowed ADR candidates"
        )

    device = allowed_candidates.device

    # Remove duplicates from strict allowed set.
    allowed_candidates = torch.unique(
        allowed_candidates
    )

    # ----------------------------------------------
    # Hard component
    # ----------------------------------------------

    selected_hard = torch.empty(
        0,
        dtype=torch.long,
        device=device,
    )

    if (
        hard_candidates is not None
        and hard_candidates.numel() > 0
        and hard_count > 0
    ):
        hard_candidates = torch.unique(
            hard_candidates
        )

        # Defensive intersection with allowed set.
        allowed_mask = torch.zeros(
            int(
                max(
                    allowed_candidates.max().item(),
                    hard_candidates.max().item(),
                )
            ) + 1,
            dtype=torch.bool,
            device=device,
        )

        allowed_mask[
            allowed_candidates
        ] = True

        hard_candidates = hard_candidates[
            allowed_mask[
                hard_candidates
            ]
        ]

        actual_hard = min(
            hard_count,
            int(hard_candidates.numel()),
        )

        if actual_hard > 0:
            perm = torch.randperm(
                hard_candidates.numel(),
                device=device,
            )

            selected_hard = (
                hard_candidates[
                    perm[:actual_hard]
                ]
            )

    actual_hard = int(
        selected_hard.numel()
    )

    # ----------------------------------------------
    # Random component
    #
    # Exclude already selected hard negatives so that
    # the realized hard/random composition is explicit.
    # ----------------------------------------------

    remaining = (
        num_negative - actual_hard
    )

    selected_random = torch.empty(
        0,
        dtype=torch.long,
        device=device,
    )

    if remaining > 0:

        # Random negatives must come from outside the
        # complete ACHNS hard pool, not merely outside the
        # selected hard subset. This makes the realized
        # composition exactly:
        #
        #   H hard + (num_negative - H) non-hard random
        #
        # whenever enough non-hard candidates exist.

        if (
            hard_candidates is not None
            and hard_candidates.numel() > 0
        ):
            keep = (
                allowed_candidates.unsqueeze(1)
                != hard_candidates.unsqueeze(0)
            ).all(dim=1)

            random_candidates = (
                allowed_candidates[keep]
            )
        else:
            random_candidates = (
                allowed_candidates
            )

        # Normally the ADR candidate space is much larger
        # than num_negative, so sample without replacement.
        if random_candidates.numel() >= remaining:
            perm = torch.randperm(
                random_candidates.numel(),
                device=device,
            )

            selected_random = (
                random_candidates[
                    perm[:remaining]
                ]
            )

        else:
            # Defensive fallback for extremely small candidate
            # sets. First use all distinct remaining candidates.
            selected_random = random_candidates

            missing = (
                remaining
                - int(selected_random.numel())
            )

            if missing > 0:
                fallback_source = (
                    random_candidates
                    if random_candidates.numel() > 0
                    else allowed_candidates
                )

                extra = (
                    _sample_from_allowed_candidates(
                        fallback_source,
                        missing,
                    )
                )

                selected_random = torch.cat(
                    [
                        selected_random,
                        extra,
                    ]
                )

    output = torch.cat(
        [
            selected_hard,
            selected_random,
        ]
    )

    if output.numel() != num_negative:
        raise RuntimeError(
            "ACHNS produced %d negatives; expected %d"
            % (
                output.numel(),
                num_negative,
            )
        )

    return output



def adr_constrained_negative_sampling(data, batch, num_negative, strict=True, task_cfg=None):
    negative_batch = _default_negative_sampling(data, batch, num_negative, strict=strict)

    adr_ids_path = _cfg_get(task_cfg, "adr_ids_path")
    relation2id_path = _cfg_get(task_cfg, "relation2id_path")
    target_relation = _cfg_get(task_cfg, "target_relation", "引起")

    achns_negative_sampling = bool(
        _cfg_get(
            task_cfg,
            "achns_negative_sampling",
            False,
        )
    )

    achns_hard_count = int(
        _cfg_get(
            task_cfg,
            "achns_hard_count",
            8,
        )
    )

    achns_pool_size = int(
        _cfg_get(
            task_cfg,
            "achns_pool_size",
            32,
        )
    )

    pp_adr_negative_sampling = bool(_cfg_get(task_cfg, "pp_adr_negative_sampling", False))
    protected_tails_path = _cfg_get(task_cfg, "protected_tails_path")
    evimoe_hard_sampling = bool(
        _cfg_get(
            task_cfg,
            "evimoe_hard_negative_sampling",
            False,
        )
    )
    evimoe_hard_fraction = float(
        _cfg_get(
            task_cfg,
            "evimoe_hard_negative_fraction",
            0.5,
        )
    )
    evimoe_exact_hard_fraction = float(
        _cfg_get(
            task_cfg,
            "evimoe_exact_positive_hard_fraction",
            evimoe_hard_fraction,
        )
    )
    evimoe_sparse_hard_fraction = float(
        _cfg_get(
            task_cfg,
            "evimoe_sparse_positive_hard_fraction",
            evimoe_hard_fraction,
        )
    )

    for value, name in (
        (
            evimoe_hard_fraction,
            "evimoe_hard_negative_fraction",
        ),
        (
            evimoe_exact_hard_fraction,
            "evimoe_exact_positive_hard_fraction",
        ),
        (
            evimoe_sparse_hard_fraction,
            "evimoe_sparse_positive_hard_fraction",
        ),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "task.%s must be between 0 and 1"
                % name
            )
    if not adr_ids_path or not relation2id_path:
        raise ValueError(
            "ADR negative sampling requires task.adr_ids_path and task.relation2id_path"
        )

    if achns_negative_sampling:
        if achns_hard_count < 0:
            raise ValueError(
                "task.achns_hard_count must be >= 0"
            )

        if achns_hard_count > num_negative:
            raise ValueError(
                "task.achns_hard_count cannot exceed "
                "task.num_negative"
            )

        if achns_pool_size <= 0:
            raise ValueError(
                "task.achns_pool_size must be > 0"
            )
    if pp_adr_negative_sampling and not protected_tails_path:
        raise ValueError(
            "PP-ADRNeg requires task.protected_tails_path when task.pp_adr_negative_sampling is enabled"
        )

    adr_candidate_ids = _load_adr_candidate_ids(adr_ids_path, batch.device, data.num_nodes)
    target_relation_id = _load_target_relation_id(relation2id_path, target_relation)
    protected_tails_by_head = (
        _load_protected_tails(protected_tails_path, batch.device)
        if pp_adr_negative_sampling
        else {}
    )
    evimoe_hard_pools = (
        _build_evimoe_ingredient_hard_pools(
            data,
            target_relation_id,
            contains_relation_id=0,
        )
        if evimoe_hard_sampling
        else {}
    )

    achns_hard_pools = (
        _build_achns_hard_pools(
            data=data,
            target_relation_id=target_relation_id,
            adr_candidate_ids=adr_candidate_ids,
            pool_size=achns_pool_size,
        )
        if achns_negative_sampling
        else {}
    )

    pos_h_index, pos_t_index, pos_r_index = batch.t()
    target_mask = pos_r_index == target_relation_id
    if not target_mask.any():
        return negative_batch

    for row_id in target_mask.nonzero(as_tuple=False).flatten():
        head_id = pos_h_index[row_id]
        true_tail_id = pos_t_index[row_id]

        allowed_mask = adr_candidate_ids != true_tail_id
        if strict:
            true_tails = _known_true_tails(data, head_id, target_relation_id)
            if true_tails.numel() > 0:
                allowed_mask &= (adr_candidate_ids.unsqueeze(1) != true_tails.unsqueeze(0)).all(dim=1)

        adr_allowed_candidates = adr_candidate_ids[allowed_mask]
        allowed_candidates = adr_allowed_candidates

        if pp_adr_negative_sampling:
            protected_tails = protected_tails_by_head.get(int(head_id.item()))
            if protected_tails is not None and protected_tails.numel() > 0:
                pp_allowed_mask = allowed_mask & (
                    adr_candidate_ids.unsqueeze(1) != protected_tails.unsqueeze(0)
                ).all(dim=1)
                pp_allowed_candidates = adr_candidate_ids[pp_allowed_mask]
                if pp_allowed_candidates.numel() >= num_negative:
                    allowed_candidates = pp_allowed_candidates

        if allowed_candidates.numel() == 0:
            allowed_candidates = adr_candidate_ids[adr_candidate_ids != true_tail_id]
        if allowed_candidates.numel() == 0:
            allowed_candidates = adr_candidate_ids

        hard_candidates = None
        selected_hard_fraction = (
            evimoe_hard_fraction
        )

        achns_hard_candidates = None

        if achns_negative_sampling:
            raw_achns_pool = (
                achns_hard_pools.get(
                    int(true_tail_id.item())
                )
            )

            if (
                raw_achns_pool is not None
                and raw_achns_pool.numel() > 0
            ):
                allowed_node_mask = torch.zeros(
                    data.num_nodes,
                    dtype=torch.bool,
                    device=batch.device,
                )

                allowed_node_mask[
                    allowed_candidates
                ] = True

                achns_hard_candidates = (
                    raw_achns_pool[
                        allowed_node_mask[
                            raw_achns_pool
                        ]
                    ]
                )

        if evimoe_hard_sampling:
            head_hard_pool = (
                evimoe_hard_pools.get(
                    int(head_id.item())
                )
            )

            (
                selected_hard_fraction,
                _,
            ) = _select_evimoe_hard_fraction(
                hard_pool=head_hard_pool,
                true_tail_id=int(
                    true_tail_id.item()
                ),
                exact_positive_fraction=
                    evimoe_exact_hard_fraction,
                sparse_positive_fraction=
                    evimoe_sparse_hard_fraction,
            )

            if (
                head_hard_pool is not None
                and head_hard_pool.numel() > 0
            ):
                allowed_node_mask = torch.zeros(
                    data.num_nodes,
                    dtype=torch.bool,
                    device=batch.device,
                )
                allowed_node_mask[
                    allowed_candidates
                ] = True
                hard_candidates = head_hard_pool[
                    allowed_node_mask[
                        head_hard_pool
                    ]
                ]

        if achns_negative_sampling:
            neg_tails = (
                _sample_achns_mixed_negatives(
                    allowed_candidates=
                        allowed_candidates,
                    hard_candidates=
                        achns_hard_candidates,
                    num_negative=num_negative,
                    hard_count=
                        achns_hard_count,
                )
            )
        elif evimoe_hard_sampling:
            neg_tails = (
                _sample_evimoe_mixed_negatives(
                    allowed_candidates=
                        allowed_candidates,
                    hard_candidates=
                        hard_candidates,
                    num_negative=num_negative,
                    hard_fraction=
                        selected_hard_fraction,
                )
            )
        else:
            neg_tails = (
                _sample_from_allowed_candidates(
                    allowed_candidates,
                    num_negative,
                )
            )

        negative_batch[row_id, :, 0] = head_id
        negative_batch[row_id, :, 1] = true_tail_id
        negative_batch[row_id, :, 2] = target_relation_id
        negative_batch[row_id, 1:, 1] = neg_tails

    return negative_batch

def negative_sampling(data, batch, num_negative, strict=True, task_cfg=None):
    if _adr_negative_sampling_enabled(task_cfg):
        return adr_constrained_negative_sampling(
            data,
            batch,
            num_negative,
            strict=strict,
            task_cfg=task_cfg,
        )
    return _default_negative_sampling(data, batch, num_negative, strict=strict)


def _default_negative_sampling(data, batch, num_negative, strict=True):
    batch_size = len(batch)
    pos_h_index, pos_t_index, pos_r_index = batch.t()

    # strict negative sampling vs random negative sampling
    if strict:
        t_mask, h_mask = strict_negative_mask(data, batch)
        t_mask = t_mask[:batch_size // 2]
        neg_t_candidate = t_mask.nonzero()[:, 1]
        num_t_candidate = t_mask.sum(dim=-1)
        # draw samples for negative tails
        rand = torch.rand(len(t_mask), num_negative, device=batch.device)
        index = (rand * num_t_candidate.unsqueeze(-1)).long()
        index = index + (num_t_candidate.cumsum(0) - num_t_candidate).unsqueeze(-1)
        neg_t_index = neg_t_candidate[index]

        h_mask = h_mask[batch_size // 2:]
        neg_h_candidate = h_mask.nonzero()[:, 1]
        num_h_candidate = h_mask.sum(dim=-1)
        # draw samples for negative heads
        rand = torch.rand(len(h_mask), num_negative, device=batch.device)
        index = (rand * num_h_candidate.unsqueeze(-1)).long()
        index = index + (num_h_candidate.cumsum(0) - num_h_candidate).unsqueeze(-1)
        neg_h_index = neg_h_candidate[index]
    else:
        neg_index = torch.randint(data.num_nodes, (batch_size, num_negative), device=batch.device)
        neg_t_index, neg_h_index = neg_index[:batch_size // 2], neg_index[batch_size // 2:]

    h_index = pos_h_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index = pos_t_index.unsqueeze(-1).repeat(1, num_negative + 1)
    r_index = pos_r_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index[:batch_size // 2, 1:] = neg_t_index
    h_index[batch_size // 2:, 1:] = neg_h_index

    return torch.stack([h_index, t_index, r_index], dim=-1)
def all_negative(data, batch):
    pos_h_index, pos_t_index, pos_r_index = batch.t()
    r_index = pos_r_index.unsqueeze(-1).expand(-1, data.num_nodes)
    # generate all negative tails for this batch
    all_index = torch.arange(data.num_nodes, device=batch.device)
    h_index, t_index = torch.meshgrid(pos_h_index, all_index)
    t_batch = torch.stack([h_index, t_index, r_index], dim=-1)
    # generate all negative heads for this batch
    all_index = torch.arange(data.num_nodes, device=batch.device)
    t_index, h_index = torch.meshgrid(pos_t_index, all_index)
    h_batch = torch.stack([h_index, t_index, r_index], dim=-1)

    return t_batch, h_batch


def strict_negative_mask(data, batch):
    # this function makes sure that for a given (h, r) batch we will NOT sample true tails as random negatives
    # similarly, for a given (t, r) we will NOT sample existing true heads as random negatives

    pos_h_index, pos_t_index, pos_r_index = batch.t()

    # part I: sample hard negative tails
    # edge index of all (head, relation) edges from the underlying graph
    edge_index = torch.stack([data.edge_index[0], data.edge_type])
    # edge index of current batch (head, relation) for which we will sample negatives
    query_index = torch.stack([pos_h_index, pos_r_index])
    # search for all true tails for the given (h, r) batch
    edge_id, num_t_truth = edge_match(edge_index, query_index)
    # build an index from the found edges
    t_truth_index = data.edge_index[1, edge_id]
    sample_id = torch.arange(len(num_t_truth), device=batch.device).repeat_interleave(num_t_truth)
    t_mask = torch.ones(len(num_t_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    # assign 0s to the mask with the found true tails
    t_mask[sample_id, t_truth_index] = 0
    t_mask.scatter_(1, pos_t_index.unsqueeze(-1), 0)

    # part II: sample hard negative heads
    # edge_index[1] denotes tails, so the edge index becomes (t, r)
    edge_index = torch.stack([data.edge_index[1], data.edge_type])
    # edge index of current batch (tail, relation) for which we will sample heads
    query_index = torch.stack([pos_t_index, pos_r_index])
    # search for all true heads for the given (t, r) batch
    edge_id, num_h_truth = edge_match(edge_index, query_index)
    # build an index from the found edges
    h_truth_index = data.edge_index[0, edge_id]
    sample_id = torch.arange(len(num_h_truth), device=batch.device).repeat_interleave(num_h_truth)
    h_mask = torch.ones(len(num_h_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    # assign 0s to the mask with the found true heads
    h_mask[sample_id, h_truth_index] = 0
    h_mask.scatter_(1, pos_h_index.unsqueeze(-1), 0)

    return t_mask, h_mask


def compute_ranking(pred, target, mask=None):
    pos_pred = pred.gather(-1, target.unsqueeze(-1))
    if mask is not None:
        # filtered ranking
        ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
    else:
        # unfiltered ranking
        ranking = torch.sum(pos_pred <= pred, dim=-1) + 1
    return ranking
