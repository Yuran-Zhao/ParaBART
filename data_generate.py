import os
from tqdm import tqdm
from xml.dom.minidom import parse
import xml.dom.minidom
import json
from multiprocessing import Pool
from collections import defaultdict
import h5py
import pdb
from transformers import BartTokenizer


class Graph():
    def __init__(self, vertices_num, graph):
        self.V = vertices_num
        # self.graph = [[0 for column in range(vertices_num)]
        #               for row in range(vertices_num)]
        # for i, js in lines.items():
        #     for j in js:
        #         self.graph[int(i)][int(j)] = 1
        self.graph = graph

    def printSolution(self, dist):
        print("Vertex tDistance from Source")
        for node in range(self.V):
            print(node, "t", dist[node])

    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree
    def minDistance(self, dist, sptSet):
        # Initialize minimum distance for next node
        min = float('inf')
        # Search not nearest vertex not in the
        # shortest path tree
        for v in range(self.V):
            if dist[v] < min and sptSet[v] == False:
                min = dist[v]
                min_index = v
        return min_index

    # Funtion that implements Dijkstra's single source
    # shortest path algorithm for a graph represented
    # using adjacency matrix representation
    def dijkstra(self, src):
        dist = [float('inf')] * self.V
        dist[src] = 0
        sptSet = [False] * self.V
        for cout in range(self.V):
            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # u is always equal to src in first iteration
            u = self.minDistance(dist, sptSet)
            # Put the minimum distance vertex in the
            # shortest path tree
            sptSet[u] = True
            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shortest path tree
            for v in range(self.V):
                if self.graph[u][v] > 0 and sptSet[
                        v] == False and dist[v] > dist[u] + self.graph[u][v]:
                    dist[v] = dist[u] + self.graph[u][v]
        # self.printSolution(dist)
        return dist


# def calculate_shortest_jump(lines):
#     for i, js in lines.items():
#         for j in js:
#             if lines.get(str(j), None) is None:
#                 lines[str(j)] = [int(i)]
#             else:
#                 lines[str(j)].append(int(i))
#     vertices_num = max([int(key) for key in lines.keys()]) + 1
#     graph = Graph(vertices_num, lines)
#     dist = [[0 for _ in range(vertices_num)] for _ in range(vertices_num)]
#     for src in range(vertices_num):
#         dist_list = graph.dijkstra(src)
#         for tgt, d in enumerate(dist_list):
#             dist[src][tgt] = d
#     return dist


def find_node_in_next_level(deleaf_synt_idx_to_tree_node_idx, deleaf_synt,
                            start):
    pre, post = 0, 0
    rets = []
    for i in range(start, len(deleaf_synt)):
        if post - pre == 1:
            break
        if deleaf_synt[i] == '(':
            pre += 1
            continue
        if deleaf_synt[i] == ')':
            post += 1
            continue
        if pre - post == 1:
            # rets.append(synt[i])
            rets.append(deleaf_synt_idx_to_tree_node_idx[i])
    return rets


def obtain_features(sent, synt, tokenizer):
    # discord the text word contained by `synt`
    deleaf_synt = deleaf(synt)

    linearized_syntax_tree = ' '.join(deleaf_synt)
    # construct the tree structure don't need '(' and ')'
    syntax_tree_nodes = linearized_syntax_tree.replace('(',
                                                       '').replace(')',
                                                                   '').split()

    # each word in the sent has its corresponding node in the syntax tree
    # we need to record the idx of its correspoding node.
    word_idx_to_deleaf_synt_idx = dict()
    for i, _ in enumerate(deleaf_synt):
        if i + 2 < len(deleaf_synt):
            if deleaf_synt[i] == '(' and not is_paren(
                    deleaf_synt[i + 1]) and deleaf_synt[i + 2] == ')':
                word_idx_to_deleaf_synt_idx[len(
                    word_idx_to_deleaf_synt_idx)] = i + 1

    deleaf_synt_idx_to_tree_node_idx = dict()
    for idx, s in enumerate(deleaf_synt):
        if s != '(' and s != ')':
            deleaf_synt_idx_to_tree_node_idx[idx] = len(
                deleaf_synt_idx_to_tree_node_idx)

    vertices_num = len(syntax_tree_nodes)
    adjacent_matrix = [[0 for column in range(vertices_num)]
                       for row in range(vertices_num)]
    for idx, s in enumerate(deleaf_synt):
        if s == '(':
            head = deleaf_synt_idx_to_tree_node_idx[idx + 1]
            tails = find_node_in_next_level(deleaf_synt_idx_to_tree_node_idx,
                                            deleaf_synt, idx + 2)
            for tail in tails:
                adjacent_matrix[head][tail] = 1
                adjacent_matrix[tail][head] = 1

    graph = Graph(vertices_num, adjacent_matrix)
    distance_between_tree_nodes = [[0 for _ in range(vertices_num)]
                                   for _ in range(vertices_num)]
    for src in range(vertices_num):
        dist_list = graph.dijkstra(src)
        for tgt, d in enumerate(dist_list):
            distance_between_tree_nodes[src][tgt] = d

    # i.e. the text words eliminated by `deleaf` function
    sent_words = sent.split()

    tmp = tokenizer(sent)['input_ids']
    sent_bart_tokenized = [tokenizer.convert_ids_to_tokens(idx)
                           for idx in tmp][1:-1]
    sent_bart_tokenized_length = len(sent_bart_tokenized)

    bart2word = align_different_tokenization(sent_words, sent_bart_tokenized)

    # sent_bart_tokenized_role_in_deleaf_synt = [
    #     deleaf_synt[word_idx_to_deleaf_synt_idx[i]] for i in range(len(sent_words))
    # ]
    # NOTE: here I use None to represent NOT FIND corresponding role
    sent_bart_tokenized_role_in_deleaf_synt = [
        deleaf_synt[word_idx_to_deleaf_synt_idx[bart2word[i]]]
        if bart2word[i] != -1 else None
        for i in range(sent_bart_tokenized_length)
    ]

    # NOTE: in the training stage, labeled by -1 will not contribute to the loss
    distance_between_bart_tokenized_sent = [[
        0 for _ in range(sent_bart_tokenized_length)
    ] for _ in range(sent_bart_tokenized_length)]

    for i in range(sent_bart_tokenized_length):
        for j in range(i + 1, sent_bart_tokenized_length):
            if bart2word[i] != -1 and bart2word[j] != -1:
                tree_node_i = deleaf_synt_idx_to_tree_node_idx[
                    word_idx_to_deleaf_synt_idx[bart2word[i]]]
                tree_node_j = deleaf_synt_idx_to_tree_node_idx[
                    word_idx_to_deleaf_synt_idx[bart2word[j]]]
                distance_between_bart_tokenized_sent[i][
                    j] = distance_between_tree_nodes[tree_node_i][tree_node_j]
                distance_between_bart_tokenized_sent[j][
                    i] = distance_between_tree_nodes[tree_node_j][tree_node_i]
            else:
                distance_between_bart_tokenized_sent[i][j] = -1
                distance_between_bart_tokenized_sent[j][i] = -1
            # assert distance_between_bart_tokenized_sent[i][j] >= 0
            # assert distance_between_bart_tokenized_sent[j][i] >= 0

    depth_of_bart_tokenized_sent = [
        0 for _ in range(sent_bart_tokenized_length)
    ]
    for i in range(sent_bart_tokenized_length):
        if bart2word[i] != -1:
            tree_node_i = deleaf_synt_idx_to_tree_node_idx[
                word_idx_to_deleaf_synt_idx[bart2word[i]]]
            depth_of_bart_tokenized_sent[i] = distance_between_tree_nodes[0][
                tree_node_i]
        else:
            depth_of_bart_tokenized_sent[i] = -1
        # assert depth_of_bart_tokenized_sent[i] > 0

    # return distance_between_bart_tokenized_sent, depth_of_bart_tokenized_sent, adjacent_matrix, syntax_tree_nodes, sent_bart_tokenized_role_in_deleaf_synt
    return distance_between_bart_tokenized_sent, depth_of_bart_tokenized_sent, adjacent_matrix, linearized_syntax_tree, sent_bart_tokenized_role_in_deleaf_synt


def is_paren(tok):
    return tok == ")" or tok == "("


def deleaf(tree):
    tree = tree.decode('utf-8')
    nonleaves = ''
    for w in tree.split():
        w = w.replace('(', '( ').replace(')', ' )')
        nonleaves += w + ' '
    arr = nonleaves.split()
    for n, i in enumerate(arr):
        if n + 1 < len(arr):
            tok1 = arr[n]
            tok2 = arr[n + 1]
            if not is_paren(tok1) and not is_paren(tok2):
                arr[n + 1] = ""
    nonleaves = " ".join(arr)
    return nonleaves.split()


# def get_parse_tree(xml_name):
#     DOMTree = xml.dom.minidom.parse(xml_name)
#     collection = DOMTree.documentElement

#     sentences = collection.getElementsByTagName("sentence")
#     synts = []
#     for sent in tqdm(sentences):
#         parse_tree = sent.getElementsByTagName("parse")[0].childNodes[0].data
#         parse_tree.strip().replace('\n', '')
#         synts.append(parse_tree)

#     with open(xml_name.replace('xml', 'synts'), 'w', encoding='utf-8') as fout:
#         for synt in tqdm(synts):
#             fout.write(synt + "\n")


def align_different_tokenization(sent_tok, sent_bart_tokenized):
    sent_bart_tokenized = [
        word.replace('Ġ', '') for word in sent_bart_tokenized
    ]
    # sent_bart_tokenized = ' '.join(sent_bart_tokenized).replace('Ġ', '').split()
    sent_tok_length = len(sent_tok)
    sent_bart_tokenized_length = len(sent_bart_tokenized)

    start = 0
    bart2word = [0] * sent_bart_tokenized_length
    j = 0
    base = ''
    while j < sent_bart_tokenized_length:
        if sent_bart_tokenized[j] == '':
            bart2word[j] = -1
            j += 1
            continue
        for i in range(start, sent_tok_length):
            if sent_tok[i] == sent_bart_tokenized[j]:
                # perfectly suitable
                bart2word[j] = i
                start = i + 1
                j += 1
                break
            elif sent_bart_tokenized[j] in sent_tok[i]:
                if base == '':
                    base = sent_tok[i]
                bart2word[j] = i
                base = base[len(sent_bart_tokenized[j]):]
                if base == '':
                    start = i + 1
                j += 1
                break
            else:
                print("[Error]! We got {} and {} in\n{}\n{}".format(
                    sent_tok[i], sent_bart_tokenized[j], sent_tok,
                    sent_bart_tokenized))
    return bart2word


# def obtain_dependency_parsing(sents, tokenizer, dictionary):
#     rets = []
#     nlp = stanza.Pipeline(lang='en',
#                           processors='tokenize,mwt,pos,lemma,depparse')
#     for sent in sents:
#         sent_tok = tokenizer(sent)['input_ids'][1:-1]
#         doc = nlp(sent)
#         tmp = []
#         start_idx = 0
#         for doc_sent in doc.sentences:
#             base, cnt = 0, 0
#             for word in doc_sent.words:
#                 for idx in range(start_idx, len(sent_tok)):
#                     if dictionary[sent_tok[idx]] in word.text:
#                         tmp.append()
#                 tmp[word.id] = tmp[word.head]
#                 cnt += 1
#             base = cnt


def main():
    para_data = h5py.File('data/data.h5', 'r')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    origin_sents1 = para_data['train_sents1']
    origin_sents1 = [sent.decode('utf-8') for sent in tqdm(origin_sents1)]
    origin_synts1 = para_data['train_synts1']

    origin_sents2 = para_data['train_sents2']
    origin_sents2 = [sent.decode('utf-8') for sent in tqdm(origin_sents2)]
    origin_synts2 = para_data['train_synts2']

    # synts1_dict = []
    synts1_edge = []
    sents1_distance = []
    sents1_depth = []
    synts1_node = []
    sents1_role = []

    print("Obtaining similarity scores...")
    sent2score = dict()
    repeat_number = 0
    for sent in tqdm(origin_sents1):
        tmp = sent.replace('-LRB-', '').replace('-RRB-', '').replace(
            '-RCB-', '').replace('-LCB-', '').replace('-LSB-',
                                                      '').replace('-RSB-', '')
        key = ''.join([c if c.isalpha() else '' for c in tmp])
        if key in sent2score:
            repeat_number += 1
        else:
            sent2score[key] = float('-inf')
    # print("There are {} examples are repeated".format(repeat_number))

    with open('data/para-nmt-50m/para-nmt-50m.txt', 'r',
              encoding='utf-8') as fin:
        for line in tqdm(fin):
            sent, _, score = line.rstrip('\n').split('\t')
            tmp = sent.lower().replace(' ', '')
            key = ''.join([c if c.isalpha() else '' for c in tmp])
            if key in sent2score:
                sent2score[key] = float(score)

    sents1, synts1, sents2, synts2 = [], [], [], []
    scores = []
    for idx, (sent1, synt1, sent2, synt2) in tqdm(
            enumerate(
                zip(origin_sents1, origin_synts1, origin_sents2,
                    origin_synts2))):
        tmp = sent1.replace('-LRB-', '').replace('-RRB-', '').replace(
            '-RCB-', '').replace('-LCB-', '').replace('-LSB-',
                                                      '').replace('-RSB-', '')
        key = ''.join([c if c.isalpha() else '' for c in tmp])
        if sent2score[key] == float('-inf'):
            continue
        else:
            sents1.append(sent1)
            synts1.append(synt1)
            sents2.append(sent2)
            synts2.append(synt2)
            scores.append(sent2score[key])
    print("There are {} examples are preserved".format(len(sents1)))
    assert len(sents1) == len(synts1) == len(sents2) == len(synts2) == len(
        scores)
    assert all(score > float('-inf') for score in scores)

    print("Generating several features for `sents1` and `synts1`...")
    for sent, synt in tqdm(zip(sents1, synts1)):
        # pdb.set_trace()
        word_distance, word_depth, synt_edge, synt_node, sent_role = obtain_features(
            sent, synt, tokenizer)
        synts1_edge.append(synt_edge)
        sents1_distance.append(word_distance)
        sents1_depth.append(word_depth)
        synts1_node.append(synt_node)
        sents1_role.append(sent_role)

    print("Generating several features for `sents2` and `synts2`...")
    synts2_link = []
    sents2_distance = []
    sents2_depth = []
    synts2_node = []
    sents2_role = []
    for sent, synt in tqdm(zip(sents2, synts2)):
        word_distance, word_depth, synt_edge, synt_node, sent_role = obtain_features(
            sent, synt, tokenizer)
        synts2_link.append(synt_edge)
        sents2_distance.append(word_distance)
        sents2_depth.append(word_depth)
        synts2_node.append(synt_node)
        sents2_role.append(sent_role)

    all_data = dict()

    all_data['train_sents1'] = sents1
    all_data['train_sents1_role'] = sents1_role
    all_data['train_synts1'] = synts1_node
    all_data['train_synts1_edge'] = synts1_edge
    all_data['train_sents1_distance'] = sents1_distance
    all_data['train_sents1_depth'] = sents1_depth

    all_data['train_sents2'] = sents2
    all_data['train_sents2_role'] = sents2_role
    all_data['train_synts2'] = synts2_node
    all_data['train_synts2_edge'] = synts2_link
    all_data['train_sents2_distance'] = sents2_distance
    all_data['train_sents2_depth'] = sents2_depth

    all_data['train_similarity_score'] = scores

    print("Dump all data...")
    with open('data/my_data.json', 'w') as f:
        json.dump(all_data, f)


def split():
    N = 994614
    n = 5000
    print("Loading all data...")
    with open('data/my_data.json', 'r') as f:
        all_data = json.load(f)
    train_data = dict()
    train_data['train_sents1'] = all_data['train_sents1'][:N - n]
    train_data['train_sents1_role'] = all_data['train_sents1_role'][:N - n]
    train_data['train_synts1'] = all_data['train_synts1'][:N - n]
    train_data['train_synts1_edge'] = all_data['train_synts1_edge'][:N - n]
    train_data['train_sents1_distance'] = all_data['train_sents1_distance'][:N -
                                                                            n]
    train_data['train_sents1_depth'] = all_data['train_sents1_depth'][:N - n]

    train_data['train_sents2'] = all_data['train_sents2'][:N - n]
    train_data['train_sents2_role'] = all_data['train_sents2_role'][:N - n]
    train_data['train_synts2'] = all_data['train_synts2'][:N - n]
    train_data['train_synts2_edge'] = all_data['train_synts2_edge'][:N - n]
    train_data['train_sents2_distance'] = all_data['train_sents2_distance'][:N -
                                                                            n]
    train_data['train_sents2_depth'] = all_data['train_sents2_depth'][:N - n]

    train_data['train_similarity_score'] = all_data[
        'train_similarity_score'][:N - n]

    for key in train_data.keys():
        assert len(train_data[key]) == N - n

    print("Dump the training data...")
    with open('data/train_data_with_score.json', 'w') as f:
        json.dump(train_data, f)

    valid_data = dict()
    valid_data['train_sents1'] = all_data['train_sents1'][N - n:]
    valid_data['train_sents1_role'] = all_data['train_sents1_role'][N - n:]
    valid_data['train_synts1'] = all_data['train_synts1'][N - n:]
    valid_data['train_synts1_edge'] = all_data['train_synts1_edge'][N - n:]
    valid_data['train_sents1_distance'] = all_data['train_sents1_distance'][N -
                                                                            n:]
    valid_data['train_sents1_depth'] = all_data['train_sents1_depth'][N - n:]

    valid_data['train_sents2'] = all_data['train_sents2'][N - n:]
    valid_data['train_sents2_role'] = all_data['train_sents2_role'][N - n:]
    valid_data['train_synts2'] = all_data['train_synts2'][N - n:]
    valid_data['train_synts2_edge'] = all_data['train_synts2_edge'][N - n:]
    valid_data['train_sents2_distance'] = all_data['train_sents2_distance'][N -
                                                                            n:]
    valid_data['train_sents2_depth'] = all_data['train_sents2_depth'][N - n:]

    valid_data['train_similarity_score'] = all_data['train_similarity_score'][
        N - n:]

    for key in valid_data.keys():
        assert len(valid_data[key]) == n
    print("Dump the validating data...")
    with open('data/valid_data_with_score.json', 'w') as f:
        json.dump(valid_data, f)


if __name__ == "__main__":
    main()
    # split()
