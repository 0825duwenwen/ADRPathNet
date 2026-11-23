import os

import torch
from torch_geometric.data import Data, InMemoryDataset, download_url


class IndRelLinkPredDataset(InMemoryDataset):

    urls = {
        "FB15k-237": [
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/test.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/valid.txt"
        ],
        "WN18RR": [
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/test.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/valid.txt"
        ]
    }

    def __init__(self, root, name, version, transform=None, pre_transform=None):
        self.name = name
        self.version = version
        assert name in ["FB15k-237", "WN18RR"]
        assert version in ["v1", "v2", "v3", "v4"]
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def num_relations(self):
        return int(self.data.edge_type.max()) + 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, self.version, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, self.version, "processed")

    @property
    def processed_file_names(self):
        return "data.pt"

    @property
    def raw_file_names(self):
        return [
            "train_ind.txt", "test_ind.txt", "train.txt", "valid.txt"
        ]

    def download(self):
        for url, path in zip(self.urls[self.name], self.raw_paths):
            download_path = download_url(url % self.version, self.raw_dir)
            os.rename(download_path, path)

    def process(self):
        test_files = self.raw_paths[:2]
        train_files = self.raw_paths[2:]

        inv_train_entity_vocab = {}
        inv_test_entity_vocab = {}
        inv_relation_vocab = {}
        triplets = []
        num_samples = []

        for txt_file in train_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    h_token, r_token, t_token = line.strip().split("\t")
                    if h_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[h_token] = len(inv_train_entity_vocab)
                    h = inv_train_entity_vocab[h_token]
                    if r_token not in inv_relation_vocab:
                        inv_relation_vocab[r_token] = len(inv_relation_vocab)
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[t_token] = len(inv_train_entity_vocab)
                    t = inv_train_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        for txt_file in test_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    h_token, r_token, t_token = line.strip().split("\t")
                    if h_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[h_token] = len(inv_test_entity_vocab)
                    h = inv_test_entity_vocab[h_token]
                    assert r_token in inv_relation_vocab
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[t_token] = len(inv_test_entity_vocab)
                    t = inv_test_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)
        triplets = torch.tensor(triplets)

        edge_index = triplets[:, :2].t()
        edge_type = triplets[:, 2]
        num_relations = int(edge_type.max()) + 1

        train_fact_slice = slice(None, sum(num_samples[:1]))
        test_fact_slice = slice(sum(num_samples[:2]), sum(num_samples[:3]))
        train_fact_index = edge_index[:, train_fact_slice]
        train_fact_type = edge_type[train_fact_slice]
        test_fact_index = edge_index[:, test_fact_slice]
        test_fact_type = edge_type[test_fact_slice]
        # add flipped triplets for the fact graphs
        train_fact_index = torch.cat([train_fact_index, train_fact_index.flip(0)], dim=-1)
        train_fact_type = torch.cat([train_fact_type, train_fact_type + num_relations])
        test_fact_index = torch.cat([test_fact_index, test_fact_index.flip(0)], dim=-1)
        test_fact_type = torch.cat([test_fact_type, test_fact_type + num_relations])

        train_slice = slice(None, sum(num_samples[:1]))
        valid_slice = slice(sum(num_samples[:1]), sum(num_samples[:2]))
        test_slice = slice(sum(num_samples[:3]), sum(num_samples))
        train_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=len(inv_train_entity_vocab),
                          target_edge_index=edge_index[:, train_slice], target_edge_type=edge_type[train_slice])
        valid_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=len(inv_train_entity_vocab),
                          target_edge_index=edge_index[:, valid_slice], target_edge_type=edge_type[valid_slice])
        test_data = Data(edge_index=test_fact_index, edge_type=test_fact_type, num_nodes=len(inv_test_entity_vocab),
                         target_edge_index=edge_index[:, test_slice], target_edge_type=edge_type[test_slice])

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

    def __repr__(self):
        return "%s()" % self.name
class ADRLinkPredDataset(InMemoryDataset):

    def __init__(self, root, name="ADR_CLEAN", transform=None, pre_transform=None):
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name)

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self):
        return [
            "train.txt",
            "valid.txt",
            "test.txt",
            "entity2id.json",
            "relation2id.json",
        ]

    @property
    def processed_file_names(self):
        return "data.pt"

    @property
    def num_relations(self):
        return int(self.data.edge_type.max()) + 1

    def download(self):
        pass

    def _load_mapping(self, path):
        import json

        with open(path, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        normalized = {}
        for key, value in mapping.items():
            if isinstance(value, int):
                normalized[str(key)] = value
            elif isinstance(key, str) and key.isdigit():
                normalized[str(value)] = int(key)
            else:
                raise ValueError("Unsupported mapping format in %s" % path)

        return normalized

    def _read_triples(self, path, entity2id, relation2id):
        heads = []
        tails = []
        rels = []

        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) != 3:
                    raise ValueError(
                        "Invalid triple at %s:%d. Expected head<TAB>relation<TAB>tail, got: %r"
                        % (path, line_no, line)
                    )

                head, relation, tail = parts

                if head not in entity2id:
                    raise KeyError("Unknown head entity %r at %s:%d" % (head, path, line_no))
                if tail not in entity2id:
                    raise KeyError("Unknown tail entity %r at %s:%d" % (tail, path, line_no))
                if relation not in relation2id:
                    raise KeyError("Unknown relation %r at %s:%d" % (relation, path, line_no))

                heads.append(entity2id[head])
                tails.append(entity2id[tail])
                rels.append(relation2id[relation])

        edge_index = torch.tensor([heads, tails], dtype=torch.long)
        edge_type = torch.tensor(rels, dtype=torch.long)
        return edge_index, edge_type

    def process(self):
        entity2id = self._load_mapping(os.path.join(self.raw_dir, "entity2id.json"))
        relation2id = self._load_mapping(os.path.join(self.raw_dir, "relation2id.json"))

        train_edge_index, train_edge_type = self._read_triples(
            os.path.join(self.raw_dir, "train.txt"), entity2id, relation2id
        )
        valid_edge_index, valid_edge_type = self._read_triples(
            os.path.join(self.raw_dir, "valid.txt"), entity2id, relation2id
        )
        test_edge_index, test_edge_type = self._read_triples(
            os.path.join(self.raw_dir, "test.txt"), entity2id, relation2id
        )

        num_nodes = len(entity2id)
        num_relations = len(relation2id)

        fact_edge_index = torch.cat([train_edge_index, train_edge_index.flip(0)], dim=-1)
        fact_edge_type = torch.cat([train_edge_type, train_edge_type + num_relations], dim=-1)

        train_data = Data(
            edge_index=fact_edge_index,
            edge_type=fact_edge_type,
            target_edge_index=train_edge_index,
            target_edge_type=train_edge_type,
            num_nodes=num_nodes,
        )

        valid_data = Data(
            edge_index=fact_edge_index,
            edge_type=fact_edge_type,
            target_edge_index=valid_edge_index,
            target_edge_type=valid_edge_type,
            num_nodes=num_nodes,
        )

        test_data = Data(
            edge_index=fact_edge_index,
            edge_type=fact_edge_type,
            target_edge_index=test_edge_index,
            target_edge_type=test_edge_type,
            num_nodes=num_nodes,
        )

        data_list = [train_data, valid_data, test_data]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
