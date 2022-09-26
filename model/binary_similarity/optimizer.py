import numpy as np
import tensorflow as tf
import os


class AttackNetwork:

    def __init__(self, model_dir):

        self.session = None
        self.graph = tf.get_default_graph()
        self.loadmodel(model_dir)

        self.max_lv = 2
        self.T_iterations = 2
        self.l2_reg_lambda = 0  # 0.002 #0.002 # regularization coefficient
        self.learning_rate = 0.5  # init learning_rate

        self.att_type = 1  # default targeted

        self.max_blocks = 150

    def loadmodel(self, log_dir):
        sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=25))
        checkpoint_dir = os.path.abspath(log_dir)
        saver = tf.train.import_meta_graph(os.path.join(checkpoint_dir, "model.meta"))
        tf.global_variables_initializer().run(session=sess)
        saver.restore(sess, os.path.join(checkpoint_dir, "model"))
        self.session = sess
        return

    def initialize_uninitialized(self):
        uninitialized_variables = [v for v in tf.global_variables() if v.name.split(':')[0]
                                   in set(
            [s.decode("utf-8") for s in self.session.run(tf.report_uninitialized_variables())])]
        self.session.run(tf.variables_initializer(uninitialized_variables))

    def meanField(self, input_x, input_adj, name):

        W1_tiled = tf.tile(tf.expand_dims(self.W1, 0), [tf.shape(input_x)[0], 1, 1], name=name + "_W1_tiled")
        W2_tiled = tf.tile(tf.expand_dims(self.W2, 0), [tf.shape(input_x)[0], 1, 1], name=name + "_W2_tiled")

        CONV_PARAMS_tiled = []
        for lv in range(self.max_lv):
            CONV_PARAMS_tiled.append(tf.tile(tf.expand_dims(self.CONV_PARAMS[lv], 0), [tf.shape(input_x)[0], 1, 1],
                                             name=name + "_CONV_PARAMS_tiled_" + str(lv)))

        w1xv = tf.matmul(input_x, W1_tiled, name=name + "_w1xv")
        l = tf.matmul(input_adj, w1xv, name=name + '_l_iteration' + str(1))
        out = w1xv
        for i in range(self.T_iterations - 1):
            ol = l
            lv = self.max_lv - 1
            while lv >= 0:
                with tf.name_scope('cell_' + str(lv)) as scope:
                    node_linear = tf.matmul(ol, CONV_PARAMS_tiled[lv], name=name + '_conv_params_' + str(lv))
                    if lv > 0:
                        ol = tf.nn.relu(node_linear, name=name + '_relu_' + str(lv))
                    else:
                        ol = node_linear
                lv -= 1
            out = tf.nn.tanh(w1xv + ol, name=name + "_mu_iteration" + str(i + 2))
            l = tf.matmul(input_adj, out, name=name + '_l_iteration' + str(i + 2))

        fi = tf.expand_dims(tf.reduce_sum(out, axis=1, name=name + "_y_potential_reduce_sum"), axis=1,
                            name=name + "_y_potential_expand_dims")

        graph_embedding = tf.matmul(fi, W2_tiled, name=name + '_graph_embedding')
        return graph_embedding

    def generateGraphClassificationNetwork(self, att_type):

        self.att_type = att_type

        # self.x_1_place = tf.placeholder(tf.float32, [None, None, 8], name="adv_x_1_place")

        # MOVR [0, 0, 1, 0, 1, 0, 0, 0, 0]
        # MOVC [1, 0, 1, 0, 1, 0, 0, 0, 0]
        # CALL [1, 0, 0, 1, 1, 0, 0, 0, 0]
        # ADDR [0, 0, 0, 0, 1, 1, 0, 0, 0]
        # ADDC [1, 0, 0, 0, 1, 1, 0, 0, 0]

        # Original input
        self.x_1 = self.graph.get_tensor_by_name("x_1:0")
        # BoW like vector for identifying dead blocks
        self.x_1_bag = tf.placeholder(tf.int32, [None, None, 8], name="adv_x1_bow")
        # Coefficients matrix
        self.e_coefficients = tf.Variable(tf.random_normal((1, self.max_blocks, 8), seed=0))

        # Input for gemini model
        self.x_1_input = tf.math.add(self.x_1, tf.math.multiply(self.x_1_bag, self.e_coefficients))

        """
        self.x_1 = tf.Variable(tf.zeros([1, self.max_blocks, 8]), name="x_1")  # Vettore del nodo in input 1

        self.set_var_x1 = self.x_1.assign(self.x_1_place)
        #self.x_1 = self.graph.get_tensor_by_name("x_1:0")
        """
        self.adj_1 = self.graph.get_tensor_by_name("adj_1:0")  # Matrice di adiacenza 1 (still placeholder)
        self.x_2 = self.graph.get_tensor_by_name("x_2:0")
        self.adj_2 = self.graph.get_tensor_by_name("adj_2:0")
        self.y = self.graph.get_tensor_by_name("y_:0")

        self.lenghts_1 = tf.placeholder(tf.float32, [None], name="len1")
        self.lenghts_2 = tf.placeholder(tf.float32, [None], name="len2")

        self.norms = []

        l2_loss = tf.constant(0.0)

        # -------------------------------
        #   1. MEAN FIELD COMPONENT
        # -------------------------------

        # 1. parameters for MeanField
        with tf.name_scope('parameters_MeanField'):

            # W1 is a [d,p] matrix, and p is the embedding size as explained above
            self.W1 = self.graph.get_tensor_by_name("parameters_MeanField/W1:0")
            self.norms.append(tf.norm(self.W1))

            # CONV_PARAMSi (i=1,...,n) is a [p,p] matrix. We refer to n as the embedding depth (self.max_lv)
            self.CONV_PARAMS = []
            for lv in range(self.max_lv):
                v = self.graph.get_tensor_by_name("parameters_MeanField/CONV_PARAMS_{}:0".format(lv))
                self.CONV_PARAMS.append(v)
                self.norms.append(tf.norm(v))

            # W2 is another [p,p] matrix to transform the embedding vector
            self.W2 = self.graph.get_tensor_by_name("parameters_MeanField/W2:0")
            self.norms.append(tf.norm(self.W2))

        # Mean Field
        with tf.name_scope('MeanField1'):
            # OLD: self.x_1 instead of self.x_1_input
            self.graph_embedding_1 = tf.nn.l2_normalize(
                tf.squeeze(self.meanField(self.x_1_input, self.adj_1, "MeanField1"), axis=1), axis=1, name="embedding1")

        with tf.name_scope('MeanField2'):
            self.graph_embedding_2 = tf.nn.l2_normalize(
                tf.squeeze(self.graph.get_tensor_by_name('MeanField2/MeanField2_graph_embedding:0'), axis=1),
                axis=1, name="embedding2")

        with tf.name_scope('Siamese'):
            self.cos_similarity = tf.reduce_sum(tf.multiply(self.graph_embedding_1, self.graph_embedding_2), axis=1,
                                                name="adv_cosSimilarity")

        # Regularization
        with tf.name_scope("Regularization"):
            l2_loss += tf.nn.l2_loss(self.x_1)

        # CalculateMean cross-entropy loss
        with tf.name_scope("Loss"):
            self.loss = tf.reduce_sum(tf.squared_difference(self.cos_similarity, self.y), name="adv_loss")
            self.regularized_loss = self.loss + self.l2_reg_lambda * l2_loss

        # Adv step: minimize if targeted maximize if untargeted
        with tf.name_scope("Adv_Step"):
            if self.att_type == 1:
                self.adv_step = tf.train.AdamOptimizer(name='adv_Adam', learning_rate=self.learning_rate) \
                    .minimize(loss=self.regularized_loss, var_list=[self.x_1])
            else:
                self.adv_step = tf.train.AdamOptimizer(name='adv_Adam', learning_rate=self.learning_rate) \
                    .maximize(loss=self.regularized_loss, var_list=[self.x_1])
