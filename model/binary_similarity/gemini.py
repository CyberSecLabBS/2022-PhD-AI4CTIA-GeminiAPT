import copy
import warnings

warnings.filterwarnings("ignore")
import os
import subprocess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import networkx as nx
import numpy as np
import json
from math import acos, pi
import time
from networkx.readwrite import json_graph
from dataset_creation.FunctionAnalyzerRadare import RadareFunctionAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from binary_similarity.optimizer import AttackNetwork

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

MODEL = "./binary_similarity/checkpoints"

LIMIT = 30
tx = 0.95
instr_list = ['movr', 'movc', 'call', 'addr', 'addc']

graph = ''


def padAndFilter(input_pairs, input_labels, max_num_vertices, min_num_vertices):
    output_pairs = []
    output_labels = []
    for pair, label in zip(input_pairs, input_labels):
        g1 = pair[0]
        g2 = pair[1]

        # graph 1
        adj1 = g1[0]
        nodes1 = g1[1]

        # graph 2
        adj2 = g2[0]
        nodes2 = g2[1]

        if (len(nodes1) <= max_num_vertices) and (len(nodes2) <= max_num_vertices):
            # graph 1
            pad_lenght1 = max_num_vertices - len(nodes1)
            new_node1 = np.pad(nodes1, [(0, pad_lenght1), (0, 0)], mode='constant')
            pad_lenght1 = max_num_vertices - adj1.shape[0]
            adj1_dense = np.pad(adj1.todense(), [(0, pad_lenght1), (0, pad_lenght1)], mode='constant')
            g1 = (adj1_dense, new_node1)
            adj2 = g2[0]
            nodes2 = g2[1]
            pad_lenght2 = max_num_vertices - len(nodes2)
            new_node2 = np.pad(nodes2, [(0, pad_lenght2), (0, 0)], mode='constant')
            pad_lenght2 = max_num_vertices - adj2.shape[0]
            adj2_dense = np.pad(adj2.todense(), [(0, pad_lenght2), (0, pad_lenght2)], mode='constant')
            g2 = (adj2_dense, new_node2)
            output_pairs.append([g1, g2])
            output_labels.append(label)
        else:
            # graph 1
            new_node1 = nodes1[0:max_num_vertices]
            adj1_dense = adj1.todense()[0:max_num_vertices, 0:max_num_vertices]
            g1 = (adj1_dense, new_node1)
            g2 = (adj1_dense, new_node1)
            output_pairs.append([g1, g2])
            output_labels.append(label)
    return output_pairs, output_labels


# Get CFG
def get_cfg(filename, function_name):
    # Get CFG
    analyzer = RadareFunctionAnalyzer(filename, use_symbol=True)
    functions = analyzer.analyze()
    return functions[function_name]['acfg']

class GEMINI:

    def __init__(self, model_dir):

        self.session = None
        self.graph = tf.get_default_graph()
        self.loadmodel(model_dir)

        self.x_1 = self.graph.get_tensor_by_name("x_1:0")
        self.adj_1 = self.graph.get_tensor_by_name("adj_1:0")  # Matrice di adiacenza 1 (still placeholder)
        self.x_2 = self.graph.get_tensor_by_name("x_2:0")
        self.adj_2 = self.graph.get_tensor_by_name("adj_2:0")

        self.emb = tf.nn.l2_normalize(
            tf.squeeze(self.graph.get_tensor_by_name('MeanField1/MeanField1_graph_embedding:0'), axis=1), axis=1,
            name="oute1")


    def loadmodel(self, log_dir):
        sess = sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=25))
        checkpoint_dir = os.path.abspath(log_dir)
        saver = tf.train.import_meta_graph(os.path.join(checkpoint_dir, "model.meta"))
        tf.global_variables_initializer().run(session=sess)
        saver.restore(sess, os.path.join(checkpoint_dir, "model"))
        self.session = sess
        return

    def embedd(self, matrice_input, nodi_input):

        out_embedding = self.session.run(self.emb, feed_dict={
            self.x_1: nodi_input,
            self.adj_1: matrice_input,
            self.x_2: nodi_input,
            self.adj_2: matrice_input})
        return out_embedding


    def feature_extraction(self, function_cfg):

        pairs = []
        label = []
        cfg_nodes = []
        adj = nx.adjacency_matrix(function_cfg)
        for n in function_cfg.nodes(data=True):
            f = n[1]['features']
            features = np.zeros(8)
            if len(f.keys()) > 0:
                if isinstance(f['constant'], list):
                    features[0] = len(f['constant'])
                else:
                    features[0] = f['constant']
                if isinstance(f['string'], list):
                    features[1] = len(f['string'])
                else:
                    features[1] = f['string']

                features[2] = f['transfer']  # mov
                features[3] = f['call']  # call
                features[4] = f['instruction']
                features[5] = f['arith']  # add
                features[6] = f['offspring']  # jmp
                features[7] = f['betweenness']
            cfg_nodes.append(features)
        pairs.append(((adj, cfg_nodes), (adj, cfg_nodes)))
        label.append(0)
        pairs, _ = padAndFilter(pairs, label, 150, 1)
        graph1, graph2 = zip(*pairs)
        adj, nodes = zip(*graph1)

        return adj[0], nodes[0]

    def embedd_CFGs(self, function_cfgs):
        adj_s, nodes_s = [], []

        for acfg in function_cfgs:
            adj, nodes = self.feature_extraction(acfg)
            adj_s.append(adj)
            nodes_s.append(nodes)
        embed = self.embedd(np.array(adj_s), np.array(nodes_s))
        return embed


# Initialize model
gemini = GEMINI(MODEL)


def run_gemini_model(functions_cfg):

    function_embeddings = []
    filtered_cfgs = [functions_cfg[el]['acfg'] for el in functions_cfg if functions_cfg[el]['acfg'].number_of_nodes() > 0]

    embeddings = gemini.embedd_CFGs(filtered_cfgs)

    for idx in range(len(filtered_cfgs)):
        function_embeddings.append({'num_nodes': filtered_cfgs[idx].number_of_nodes(),
                                    'embedding': embeddings[idx]})

    """
    for el in functions_cfg:
        if functions_cfg[el]['acfg'].number_of_nodes() > 0:
            function_embeddings.append()
    """

    return function_embeddings
