import torch
import torch.nn as nn
from torch.utils.data import Dataset
import json
import dgl
import random
import pdb


class StructuralDataset(Dataset):
    def __init__(self,
                 json_file,
                 tokenizer,
                 synt_vocab,
                 max_sent_len,
                 max_synt_len,
                 use_linear_tree=True,
                 bos_token="<s>",
                 eos_token="</s>",
                 pad_token="<pad>",
                 unk_token="<unk>"):
        self.tokenizer = tokenizer
        self.synt_vocab = synt_vocab
        self.max_sent_len = max_sent_len
        self.max_synt_len = max_synt_len
        self.use_linear_tree = use_linear_tree

        with open(json_file, 'r') as f:
            all_data = json.load(f)

        self.sents1 = all_data['train_sents1']
        self.sents1_role = all_data['train_sents1_role']
        self.synts1 = all_data['train_synts1']
        self.synts1_edge = all_data['train_synts1_edge']
        self.sents1_distance = all_data['train_sents1_distance']
        self.sents1_depth = all_data['train_sents1_depth']

        self.sents2 = all_data['train_sents2']
        self.sents2_role = all_data['train_sents2_role']
        self.synts2 = all_data['train_synts2']
        self.synts2_edge = all_data['train_synts2_edge']
        self.sents2_distance = all_data['train_sents2_distance']
        self.sents2_depth = all_data['train_sents2_depth']

        self.similarity_score = all_data['train_similarity_score']

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token

        # random.seed(1234)
        # random.shuffle(self.sents1)
        # random.seed(1234)
        # random.shuffle(self.sents1_role)
        # random.seed(1234)
        # random.shuffle(self.synts1)
        # random.seed(1234)
        # random.shuffle(self.synts1_edge)
        # random.seed(1234)
        # random.shuffle(self.sents1_distance)
        # random.seed(1234)
        # random.shuffle(self.sents1_depth)

        # random.seed(1234)
        # random.shuffle(self.sents2)
        # random.seed(1234)
        # random.shuffle(self.sents2_role)
        # random.seed(1234)
        # random.shuffle(self.synts2_edge)
        # random.seed(1234)
        # random.shuffle(self.synts2_edge)
        # random.seed(1234)
        # random.shuffle(self.sents2_distance)
        # random.seed(1234)
        # random.shuffle(self.sents2_depth)

        # self.data_gen = self.get_data()

    def __len__(self):
        return len(self.sents1)

    # def get_data(self):
    #     for idx in range(len(self.sents1)):
    #         sent1_token_ids = torch.ones((self.max_sent_len + 2),
    #                                      dtype=torch.long)
    #         synt1_token_ids = torch.ones((self.max_synt_len + 2),
    #                                      dtype=torch.long)
    #         sent1_target_role_ids = torch.ones((self.max_sent_len + 2),
    #                                            dtype=torch.long)
    #         sent1_distance = torch.full(
    #             (self.max_sent_len + 2, self.max_sent_len + 2),
    #             fill_value=-1,
    #             dtype=torch.long)
    #         sent1_depth = torch.full((1, self.max_sent_len + 2),
    #                                  fill_value=-1,
    #                                  dtype=torch.long)
    #         synt1_bow = torch.ones((74))

    #         sent2_token_ids = torch.ones((self.max_sent_len + 2),
    #                                      dtype=torch.long)
    #         synt2_token_ids = torch.ones((self.max_synt_len + 2),
    #                                      dtype=torch.long)
    #         sent2_target_role_ids = torch.ones((self.max_sent_len + 2),
    #                                            dtype=torch.long)
    #         sent2_distance = torch.full(
    #             (self.max_sent_len + 2, self.max_sent_len + 2),
    #             fill_value=-1,
    #             dtype=torch.long)
    #         sent2_depth = torch.full((1, self.max_sent_len + 2),
    #                                  fill_value=-1,
    #                                  dtype=torch.long)
    #         synt2_bow = torch.ones((74))

    #         sent1_inputs = self.tokenizer(self.sents1[idx],
    #                                       padding='max_length',
    #                                       truncation=True,
    #                                       max_length=self.max_sent_len + 2,
    #                                       return_tensors="pt")
    #         sent2_inputs = self.tokenizer(self.sents2[idx],
    #                                       padding='max_length',
    #                                       truncation=True,
    #                                       max_length=self.max_sent_len + 2,
    #                                       return_tensors="pt")
    #         sent1_token_ids = sent1_inputs['input_ids'].squeeze(0)
    #         sent1_length = sent1_inputs['attention_mask'].sum() - 2

    #         sent2_token_ids = sent2_inputs['input_ids'].squeeze(0)
    #         sent2_length = sent2_inputs['attention_mask'].sum() - 2

    #         if self.use_linear_tree:
    #             synt1 = [self.bos_token
    #                      ] + self.synts1[idx].split() + [self.eos_token]
    #         else:
    #             synt1 = [self.bos_token] + self.synts1[idx].replace(
    #                 '(', '').replace(')', '').split() + [self.eos_token]
    #         synt1_token_ids[:len(synt1)] = torch.tensor(
    #             [self.synt_vocab[tag] for tag in synt1])[:self.max_synt_len + 2]

    #         # sent1_target_role_ids[1:len(self.sents1_role[idx]) + 1] = torch.tensor([
    #         #     self.synt_vocab[tag] if tag != -1 else -1
    #         #     for tag in self.sents1_role[idx]
    #         # ])[:self.max_sent_len - 1]

    #         # sent1_distance[1:len(self.sents1[idx].split()) + 1,
    #         #                1:len(self.sents1[idx].split()) + 1] = torch.tensor(
    #         #                    self.sents1_distance[idx])[:self.max_sent_len, :self.
    #         #                                               max_sent_len]
    #         # sent1_depth[1:len(self.sents1[idx].split()) + 1] = torch.tensor(
    #         #     self.sents1_depth[idx])[:self.max_sent_len]

    #         sent1_distance[1:sent1_length + 1,
    #                        1:sent1_length + 1] = torch.tensor(
    #                            self.sents1_distance[idx]
    #                        )[:self.max_sent_len, :self.max_sent_len]
    #         sent1_depth[0][1:sent1_length + 1] = torch.tensor(
    #             self.sents1_depth[idx])[:self.max_sent_len]

    #         if self.use_linear_tree:
    #             synt2 = [self.bos_token
    #                      ] + self.synts2[idx].split() + [self.eos_token]
    #         else:
    #             synt2 = [self.bos_token] + self.synts2[idx].replace(
    #                 '(', '').replace(')', '').split() + [self.eos_token]

    #         synt2_token_ids[:len(synt2)] = torch.tensor(
    #             [self.synt_vocab[tag] for tag in synt2])[:self.max_synt_len + 2]

    #         # sent2_target_role_ids[1:len(self.sents2_role[idx]) + 1] = torch.tensor([
    #         #     self.synt_vocab[tag] if tag != -1 else -1
    #         #     for tag in self.sents2_role[idx]
    #         # ])[:self.max_sent_len - 1]

    #         # sent2_distance[1:len(self.sents2[idx].split()) + 1,
    #         #                1:len(self.sents2[idx].split()) + 1] = torch.tensor(
    #         #                    self.sents2_distance[idx])[:self.max_sent_len, :self.
    #         #                                               max_sent_len]
    #         # sent2_depth[1:len(self.sents2[idx].split()) + 1] = torch.tensor(
    #         #     self.sents2_depth[idx])[:self.max_sent_len]

    #         sent2_distance[1:sent2_length + 1,
    #                        1:sent2_length + 1] = torch.tensor(
    #                            self.sents2_distance[idx]
    #                        )[:self.max_sent_len, :self.max_sent_len]
    #         sent2_depth[0][1:sent2_length + 1] = torch.tensor(
    #             self.sents2_depth[idx])[:self.max_sent_len]

    #         for tag in synt1:
    #             if tag != self.bos_token and tag != self.eos_token:
    #                 synt1_bow[self.synt_vocab[tag] - 3] += 1
    #         synt1_bow /= synt1_bow.sum(0, keepdim=True)

    #         for tag in synt2:
    #             if tag != self.bos_token and tag != self.eos_token:
    #                 synt2_bow[self.synt_vocab[tag] - 3] += 1
    #         synt2_bow /= synt2_bow.sum(0, keepdim=True)

    #         # pdb.set_trace()

    #         adjacent_matrix1 = self.synts1_edge[idx]
    #         out_node1 = []
    #         in_node1 = []
    #         for i in range(len(adjacent_matrix1)):
    #             for j in range(len(adjacent_matrix1)):
    #                 if adjacent_matrix1[i][j] == 1:
    #                     out_node1.append(i)
    #                     in_node1.append(j)
    #         in_node1 = torch.LongTensor(in_node1)
    #         out_node1 = torch.LongTensor(out_node1)
    #         graph1 = dgl.graph((out_node1, in_node1),
    #                            num_nodes=len(adjacent_matrix1))

    #         adjacent_matrix2 = self.synts2_edge[idx]
    #         out_node2 = []
    #         in_node2 = []
    #         for i in range(len(adjacent_matrix2)):
    #             for j in range(len(adjacent_matrix2)):
    #                 if adjacent_matrix2[i][j] == 1:
    #                     out_node2.append(i)
    #                     in_node2.append(j)
    #         in_node2 = torch.LongTensor(in_node2)
    #         out_node2 = torch.LongTensor(out_node2)
    #         graph2 = dgl.graph((out_node2, in_node2),
    #                            num_nodes=len(adjacent_matrix2))

    #         yield sent1_token_ids, synt1_token_ids, sent1_distance, sent1_depth[
    #             0], synt1_bow, graph1, sent2_token_ids, synt2_token_ids, sent2_distance, sent2_depth[
    #                 0], synt2_bow, graph2

    # def __getitem__(self, idx):
    #     sent1_token_ids, synt1_token_ids, sent1_distance, sent1_depth, synt1_bow, graph1, sent2_token_ids, synt2_token_ids, sent2_distance, sent2_depth, synt2_bow, graph2 = next(
    #         self.data_gen)
    #     pdb.set_trace()
    #     return sent1_token_ids, synt1_token_ids, sent1_distance, sent1_depth, synt1_bow, graph1, sent2_token_ids, synt2_token_ids, sent2_distance, sent2_depth, synt2_bow, graph2

    def __getitem__(self, idx):
        sent1_token_ids = torch.ones((self.max_sent_len + 2), dtype=torch.long)
        synt1_token_ids = torch.ones((self.max_synt_len + 2), dtype=torch.long)
        sent1_target_role_ids = torch.ones((self.max_sent_len + 2),
                                           dtype=torch.long)
        sent1_distance = torch.full(
            (self.max_sent_len + 2, self.max_sent_len + 2),
            fill_value=-1,
            dtype=torch.long)
        sent1_depth = torch.full((1, self.max_sent_len + 2),
                                 fill_value=-1,
                                 dtype=torch.long)
        synt1_bow = torch.ones((74))

        sent2_token_ids = torch.ones((self.max_sent_len + 2), dtype=torch.long)
        synt2_token_ids = torch.ones((self.max_synt_len + 2), dtype=torch.long)
        sent2_target_role_ids = torch.ones((self.max_sent_len + 2),
                                           dtype=torch.long)
        sent2_distance = torch.full(
            (self.max_sent_len + 2, self.max_sent_len + 2),
            fill_value=-1,
            dtype=torch.long)
        sent2_depth = torch.full((1, self.max_sent_len + 2),
                                 fill_value=-1,
                                 dtype=torch.long)
        synt2_bow = torch.ones((74))
        similarity_score = torch.ones((1))

        sent1_inputs = self.tokenizer(self.sents1[idx],
                                      padding='max_length',
                                      truncation=True,
                                      max_length=self.max_sent_len + 2,
                                      return_tensors="pt")
        sent2_inputs = self.tokenizer(self.sents2[idx],
                                      padding='max_length',
                                      truncation=True,
                                      max_length=self.max_sent_len + 2,
                                      return_tensors="pt")
        sent1_token_ids = sent1_inputs['input_ids'].squeeze(0)
        sent1_length = sent1_inputs['attention_mask'].sum() - 2

        sent2_token_ids = sent2_inputs['input_ids'].squeeze(0)
        sent2_length = sent2_inputs['attention_mask'].sum() - 2

        if self.use_linear_tree:
            synt1 = [self.bos_token
                     ] + self.synts1[idx].split() + [self.eos_token]
        else:
            synt1 = [self.bos_token] + self.synts1[idx].replace(
                '(', '').replace(')', '').split() + [self.eos_token]
        synt1_token_ids[:len(synt1)] = torch.tensor(
            [self.synt_vocab[tag] for tag in synt1])[:self.max_synt_len + 2]

        # sent1_target_role_ids[1:len(self.sents1_role[idx]) + 1] = torch.tensor([
        #     self.synt_vocab[tag] if tag != -1 else -1
        #     for tag in self.sents1_role[idx]
        # ])[:self.max_sent_len - 1]

        # sent1_distance[1:len(self.sents1[idx].split()) + 1,
        #                1:len(self.sents1[idx].split()) + 1] = torch.tensor(
        #                    self.sents1_distance[idx])[:self.max_sent_len, :self.
        #                                               max_sent_len]
        # sent1_depth[1:len(self.sents1[idx].split()) + 1] = torch.tensor(
        #     self.sents1_depth[idx])[:self.max_sent_len]

        sent1_distance[1:sent1_length + 1, 1:sent1_length + 1] = torch.tensor(
            self.sents1_distance[idx])[:self.max_sent_len, :self.max_sent_len]
        sent1_depth[0][1:sent1_length + 1] = torch.tensor(
            self.sents1_depth[idx])[:self.max_sent_len]

        if self.use_linear_tree:
            synt2 = [self.bos_token
                     ] + self.synts2[idx].split() + [self.eos_token]
        else:
            synt2 = [self.bos_token] + self.synts2[idx].replace(
                '(', '').replace(')', '').split() + [self.eos_token]

        synt2_token_ids[:len(synt2)] = torch.tensor(
            [self.synt_vocab[tag] for tag in synt2])[:self.max_synt_len + 2]

        # sent2_target_role_ids[1:len(self.sents2_role[idx]) + 1] = torch.tensor([
        #     self.synt_vocab[tag] if tag != -1 else -1
        #     for tag in self.sents2_role[idx]
        # ])[:self.max_sent_len - 1]

        # sent2_distance[1:len(self.sents2[idx].split()) + 1,
        #                1:len(self.sents2[idx].split()) + 1] = torch.tensor(
        #                    self.sents2_distance[idx])[:self.max_sent_len, :self.
        #                                               max_sent_len]
        # sent2_depth[1:len(self.sents2[idx].split()) + 1] = torch.tensor(
        #     self.sents2_depth[idx])[:self.max_sent_len]

        sent2_distance[1:sent2_length + 1, 1:sent2_length + 1] = torch.tensor(
            self.sents2_distance[idx])[:self.max_sent_len, :self.max_sent_len]
        sent2_depth[0][1:sent2_length + 1] = torch.tensor(
            self.sents2_depth[idx])[:self.max_sent_len]

        for tag in synt1:
            if tag != self.bos_token and tag != self.eos_token:
                synt1_bow[self.synt_vocab[tag] - 3] += 1
        synt1_bow /= synt1_bow.sum(0, keepdim=True)

        for tag in synt2:
            if tag != self.bos_token and tag != self.eos_token:
                synt2_bow[self.synt_vocab[tag] - 3] += 1
        synt2_bow /= synt2_bow.sum(0, keepdim=True)

        # pdb.set_trace()

        adjacent_matrix1 = self.synts1_edge[idx]
        out_node1 = []
        in_node1 = []
        for i in range(len(adjacent_matrix1)):
            for j in range(len(adjacent_matrix1)):
                if adjacent_matrix1[i][j] == 1:
                    out_node1.append(i)
                    in_node1.append(j)
        in_node1 = torch.LongTensor(in_node1)
        out_node1 = torch.LongTensor(out_node1)
        graph1 = dgl.graph((out_node1, in_node1),
                           num_nodes=len(adjacent_matrix1))

        adjacent_matrix2 = self.synts2_edge[idx]
        out_node2 = []
        in_node2 = []
        for i in range(len(adjacent_matrix2)):
            for j in range(len(adjacent_matrix2)):
                if adjacent_matrix2[i][j] == 1:
                    out_node2.append(i)
                    in_node2.append(j)
        in_node2 = torch.LongTensor(in_node2)
        out_node2 = torch.LongTensor(out_node2)
        graph2 = dgl.graph((out_node2, in_node2),
                           num_nodes=len(adjacent_matrix2))

        similarity_score = self.similarity_score[idx]

        # return sent1_token_ids, synt1_token_ids, sent1_target_role_ids, sent1_distance, sent1_depth[
        #     0], synt1_bow, graph1, sent2_token_ids, synt2_token_ids, sent2_target_role_ids, sent2_distance, sent2_depth[
        #         0], synt2_bow, graph2

        return sent1_token_ids, synt1_token_ids, sent1_distance, sent1_depth[
            0], synt1_bow, graph1, sent2_token_ids, synt2_token_ids, sent2_distance, sent2_depth[
                0], synt2_bow, graph2, similarity_score
