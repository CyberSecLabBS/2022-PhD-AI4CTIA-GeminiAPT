import numpy as np
import argparse
import json
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
import os
import sqlite3
import tensorflow as tf


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

class S2VModel:

    def __init__(self,model_dir):
        self.logdir = model_dir
        self.session = None
        self.loadmodel()

    def loadmodel(self):
        sess = tf.Session()
        checkpoint_dir = os.path.abspath(os.path.join(self.logdir, "checkpoints"))
        saver = tf.train.import_meta_graph(os.path.join(checkpoint_dir,"model.meta"))
        tf.global_variables_initializer().run(session=sess)
        saver.restore(sess, os.path.join(checkpoint_dir, "model"))
        self.session = sess
        return

    def embedd(self, matrice_input, nodi_input):
        graph = tf.get_default_graph()
        x_1 = graph.get_tensor_by_name("x_1:0")
        adj_1 = graph.get_tensor_by_name("adj_1:0")
        x_2=graph.get_tensor_by_name("x_2:0")
        adj_2= graph.get_tensor_by_name("adj_2:0")
        emb=tf.nn.l2_normalize(tf.squeeze(graph.get_tensor_by_name('MeanField1/MeanField1_graph_embedding:0'), axis=1), axis=1,name="oute1")
        out_embedding= self.session.run(emb,feed_dict = {
                        x_1: nodi_input,
                        adj_1: matrice_input,
                        x_2: nodi_input,
                        adj_2: matrice_input})
        return out_embedding

    def close(self):
        print("Closing tensorflow session")
        self.session.close()


class FunctionEmbedder:

    def __init__(self,s2v_network, db_name, table_name):
        self.s2v_network=s2v_network
        self.db_name = db_name
        self.table_name = table_name

    def embeddCFG(self, cfgs):
        pairs = []
        label = []
        for cfg in cfgs:
            cfg_nodes = []
            adj = nx.adjacency_matrix(cfg)
            for n in cfg.nodes(data=True):
                f = n[1]['features']
                features=np.zeros(8)
                if len(f.keys()) > 0:
                    if isinstance(f['constant'], list):
                        features[0] = len(f['constant'])
                    else:
                        features[0] = f['constant']
                    if isinstance(f['string'], list):
                        features[1] = len(f['string'])
                    else:
                        features[1] = f['string']

                    features[2] = f['transfer']
                    features[3] = f['call']
                    features[4] = f['instruction']
                    features[5] = f['arith']
                    features[6] = f['offspring']
                    features[7] = f['betweenness']
                cfg_nodes.append(features)
            pairs.append(((adj, cfg_nodes), (adj, cfg_nodes)))
            label.append(0)

        pairs, _ = padAndFilter(pairs, label, 150, 1)
        graph1, graph2 = zip(*pairs)
        adj, nodes = zip(*graph1)
        embed=s2v.embedd(adj, nodes)
        return embed

    def create_table(self):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS {} (id INTEGER PRIMARY KEY, {}  TEXT)".format(self.table_name, self.table_name))
        conn.commit()
        conn.close()

    def embedd_and_save(self):
        self.create_table()

        conn = sqlite3.connect(self.db_name)
        cur = conn.cursor()
        q = cur.execute("SELECT id FROM acfg WHERE id NOT IN (SELECT id FROM gemini)")
        ids = q.fetchall()
        ids = [i[0] for i in ids]
        print("Selected " + str(len(ids)) + " functions")

        batch_size = 512
        for j in range(0,len(ids),batch_size):
            batch_ids = []
            cfgs = []
            for k in range(j, j+batch_size):
                if k == len(ids):
                    break
                q = cur.execute("SELECT acfg FROM acfg WHERE id=?", (ids[k],))
                fetched = q.fetchone()
                if fetched is  None:
                    continue
                c = json_graph.adjacency_graph(json.loads(fetched[0]))
                batch_ids.append(ids[k])
                cfgs.append(c)
            e = self.embeddCFG(cfgs)
            print("Done Batch from " + str(j) + " to " + str(j+batch_size))

            for l, id in enumerate(batch_ids):
                if l < e.shape[0]:
                    cur.execute("INSERT INTO " + self.table_name + " VALUES (?,?)",(id, np.array2string(e[l])))
            conn.commit()
        conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-db", "--db_name", dest="db_name", help="Databases with function to embed",required=True)
    parser.add_argument("-tb", "--table_name", dest="table_name", help="Table name to store embedding",required=True)
    parser.add_argument("-m",  "--model", dest="model", help="Model to use for embedding creation")

    args = parser.parse_args()

    s2v=S2VModel(args.model)
    s2v.loadmodel()

    fe=FunctionEmbedder(s2v, args.db_name, args.table_name)
    fe.embedd_and_save()

    s2v.close()

'''
vi allego uno script per calcolare gli embedding con gemini. Nel file zip trovate anche il modello trainato.
Lo script si aspetta di leggere i grafi delle funzioni in formato json da un db ma ovviamente potete adattarlo al vostro caso.
La cosa importante è che guardiate l
'''
