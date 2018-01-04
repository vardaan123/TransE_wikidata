import argparse
import math
import os.path
import timeit
import codecs
import numpy as np
import tensorflow as tf
from load_wikidata2 import load_wikidata
import random
from multiprocessing import JoinableQueue, Queue, Process
from annoy import AnnoyIndex

class TransE:
    @property
    def n_entity(self):
        return self.__n_entity

    @property
    def n_train(self):
        return self.__train_triple.shape[0]

    @property
    def embed_dim(self):
        return self.__embed_dim

    @property
    def eval_batch(self):
        return self.eval_batch

    # @property
    # def trainable_variables(self):
    #     return self.__trainable

    @property
    def hr_t(self):
        return self.__hr_t

    @property
    def tr_h(self):
        return self.__tr_h

    @property
    def ent_embedding(self):
        return self.ent_embeddings

    @property
    def rel_embedding(self):
        return self.__rel_embedding

    @property
    def n_relation(self):
        return self.__n_relation

    def raw_training_data(self, batch_size=100):
        n_triple = len(self.__train_triple)
        rand_idx = np.random.permutation(n_triple)

        start = 0
        while start < n_triple:
            end = min(start + batch_size, n_triple)
            yield self.__train_triple[rand_idx[start:end]]
            start = end

    def testing_data(self, batch_size=100):
        n_triple = len(self.__test_triple)
        start = 0
        while start < n_triple:
            end = min(start + batch_size, n_triple)
            yield self.__test_triple[start:end, :]
            start = end

    def validation_data(self, batch_size=100):
        n_triple = len(self.__valid_triple)
        start = 0
        while start < n_triple:
            end = min(start + batch_size, n_triple)
            yield self.__valid_triple[start:end, :]
            start = end

    def corrupt_sample(self,htr):
        neg_h_list = []
        neg_t_list = []
        for idx in range(htr.shape[0]):
            prob = float(np.random.sample())
            #print prob
            if(prob < 0.5):
                neg_t = -1
                while neg_t < 0:
                    tmp = np.random.randint(0,self.__n_entity)
                    if tmp not in self.__hr_t[htr[idx, 0]][htr[idx, 2]]:
                        neg_t = tmp
                        #print "before ", neg_t_list[idx]
                        neg_t_list.append(neg_t)
                        neg_h_list.append(htr[idx,0])
                        #print "after ", neg_t_list[idx]
            else:            
                neg_h = -1
                while neg_h < 0:
                    tmp = np.random.randint(0,self.__n_entity)
                    if tmp not in self.__tr_h[htr[idx, 1]][htr[idx, 2]]:
                        neg_h = tmp
                        neg_h_list.append(neg_h)
                        neg_t_list.append(htr[idx,1])
        neg_h_list = np.asarray(neg_h_list,dtype=np.int32)
        neg_t_list = np.asarray(neg_t_list,dtype=np.int32)                
        return htr[:,0] , htr[:,2], htr[:, 1], neg_h_list, htr[:,2], neg_t_list

    def __init__(self, data_dir, embed_dim=100, fanout_thresh=2,eval_batch=32):

    
        self.__embed_dim = embed_dim
        self.__initialized = True
        self.eval_batch = eval_batch

        # self.__trainable = list()

        # ********************************************************************************************
        with tf.device('/cpu'):
            wikidata, prop_data, wikidata_fanout_dict, child_par_dict = load_wikidata()
            # print len(wikidata)
            wikidata_remove_list = [q for q in wikidata if wikidata_fanout_dict[q] <= fanout_thresh]

            if fanout_thresh == 2:
                wikidata_remove_list.extend(wikidata.keys()[-100000:])
                
            for q in wikidata_remove_list:
                wikidata.pop(q, None)

            self.__relation_id_map = {pid:i for i,pid in enumerate(prop_data.keys())}
            self.__entity_id_map = {qid:i for i,qid in enumerate(wikidata.keys())}

            self.__id_relation_map = {i:pid for i,pid in enumerate(prop_data.keys())}
            self.__id_entity_map = {i:qid for i,qid in enumerate(wikidata.keys())}

            self.__n_entity = len(self.__entity_id_map.keys())
            self.__n_relation = len(self.__relation_id_map.keys())


            def load_triple():
                triples_arr = []

                for QID in self.__entity_id_map.keys():
                    for pid in [p for p in wikidata[QID] if p in self.__relation_id_map]:
                        for qid in [q for q in wikidata[QID][pid] if q in child_par_dict and q in self.__entity_id_map]:
                            triples_arr.append([self.__entity_id_map[QID], self.__entity_id_map[qid], self.__relation_id_map[pid]])
                            # if len(triples_arr) > 10000:
                            #     return np.asarray(triples_arr,dtype=np.int32)
                return np.asarray(triples_arr,dtype=np.int32)

            # self.__n_entity = 2900000
            # self.__n_relation = 567


            # def load_triple():
            #     triples_arr = []

            #     for i in xrange(11963): #11963105
            #         triples_arr.append([random.randint(0,self.__n_entity-1), random.randint(0,self.__n_entity-1), random.randint(0,self.__n_relation-1)])
            #                 # if len(triples_arr) > 1000000:
            #                 #     return np.asarray(triples_arr)
            #     return np.asarray(triples_arr, dtype=np.int32)

            triples_arr = load_triple()
            idx = np.random.permutation(np.arange(triples_arr.shape[0]))

            self.__train_triple = triples_arr
            self.__valid_triple = np.array([])
            self.__test_triple = np.array([])

        # ********************************************************************************************
        # with codecs.open(os.path.join(data_dir, 'entity2id.txt'), 'r', encoding='utf-8') as f:
        #     self.__n_entity = len(f.readlines())

        # with codecs.open(os.path.join(data_dir, 'entity2id.txt'), 'r', encoding='utf-8') as f:
        #     self.__entity_id_map = {x.strip().split('\t')[0]: int(x.strip().split('\t')[1]) for x in f.readlines()}
        #     self.__id_entity_map = {v: k for k, v in self.__entity_id_map.items()}

        # with codecs.open(os.path.join(data_dir, 'relation2id.txt'), 'r', encoding='utf-8') as f:
        #     self.__n_relation = len(f.readlines())
        

        # with codecs.open(os.path.join(data_dir, 'relation2id.txt'), 'r', encoding='utf-8') as f:
        #     self.__relation_id_map = {x.strip().split('\t')[0]: int(x.strip().split('\t')[1]) for x in f.readlines()}
        #     self.__id_relation_map = {v: k for k, v in self.__entity_id_map.items()}
        
        # def load_triple(file_path):
        #     with codecs.open(file_path, 'r', encoding='utf-8') as f_triple:
        #         return np.asarray([[self.__entity_id_map[x.strip().split('\t')[0]],
        #                             self.__entity_id_map[x.strip().split('\t')[1]],
        #                             self.__relation_id_map[x.strip().split('\t')[2]]] for x in f_triple.readlines()],
        #                           dtype=np.int32)
        
        # self.__train_triple = load_triple(os.path.join(data_dir, 'train.txt'))
        # self.__test_triple = load_triple(os.path.join(data_dir, 'test.txt'))
        # self.__valid_triple = load_triple(os.path.join(data_dir, 'valid.txt'))

        # ********************************************************************************************
            print("N_ENTITY: %d" % self.__n_entity)
            print("N_RELATION: %d" % self.__n_relation)

            print("N_TRAIN_TRIPLES: %d" % self.__train_triple.shape[0])
            
            # print("N_TEST_TRIPLES: %d" % self.__test_triple.shape[0])

            # print("N_VALID_TRIPLES: %d" % self.__valid_triple.shape[0])

            def gen_hr_t(triple_data):
                hr_t = dict()
                for h, t, r in triple_data:
                    if h not in hr_t:
                        hr_t[h] = dict()
                    if r not in hr_t[h]:
                        hr_t[h][r] = set()
                    hr_t[h][r].add(t)
                return hr_t

            def gen_tr_h(triple_data):
                tr_h = dict()
                for h, t, r in triple_data:
                    if t not in tr_h:
                        tr_h[t] = dict()
                    if r not in tr_h[t]:
                        tr_h[t][r] = set()
                    tr_h[t][r].add(h)
                return tr_h

            #self.__train_hr_t = gen_hr_t(self.__train_triple)
            #self.__train_tr_h = gen_tr_h(self.__train_triple)
            #self.__test_hr_t = gen_hr_t(self.__test_triple)
            #self.__test_tr_h = gen_tr_h(self.__test_triple)

            self.__hr_t = gen_hr_t(self.__train_triple)
            self.__tr_h = gen_tr_h(self.__train_triple)

        bound = 6 / math.sqrt(embed_dim)

        # with tf.device('/cpu'):
        self.ent_embeddings = tf.get_variable("ent_embedding", [self.__n_entity, embed_dim],
                                                   initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                             maxval=bound,
                                                                                             seed=345))
            # self.__trainable.append(self.ent_embeddings)

        self.rel_embeddings = tf.get_variable("rel_embedding", [self.__n_relation, embed_dim],
                                                   initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                             maxval=bound,
                                                                                             seed=346))
            # self.__trainable.append(self.rel_embeddings)

    @staticmethod
    def __l1_normalize(x, dim, epsilon=1e-12, name=None):
        square_sum = tf.reduce_sum(tf.abs(x), [dim], keep_dims=True)
        x_inv_norm = tf.rsqrt(tf.maximum(square_sum, epsilon))
        return tf.mul(x, x_inv_norm, name=name)
    
    def train(self, inputs, regularizer_weight=1., scope=None):
        with tf.variable_scope(scope or type(self).__name__) as scp:
            if self.__initialized:
                scp.reuse_variables()

            pos_neg_list = inputs

            # pos_list = tf.slice(pos_neg_list, [0, 0], [-1, 3])
            # neg_list = tf.slice(pos_neg_list, [0, 3], [-1, 3])

            # pos_list_embed = tf.nn.embedding_lookup([self.ent_embeddings, self.rel_embeddings, self.ent_embeddings], pos_list, partition_strategy='mod')
            # neg_list_embed = tf.nn.embedding_lookup([self.ent_embeddings, self.rel_embeddings, self.ent_embeddings], neg_list, partition_strategy='mod')

            # pos_list_score = tf.reduce_sum((tf.sub(tf.reduce_sum(tf.slice(pos_list_embed,[0, 0, 0],[-1, 2, -1]),1), tf.slice(pos_list_embed,[0, 2, 0],[-1, 1, -1])))**2)
            # neg_list_score = tf.reduce_sum((tf.sub(tf.reduce_sum(tf.slice(neg_list_embed,[0, 0, 0],[-1, 2, -1]),1), tf.slice(neg_list_embed,[0, 2, 0],[-1, 1, -1])))**2)
           
            # # pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keep_dims = True)
            # # neg = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keep_dims = True)
            # loss = tf.reduce_sum(tf.maximum(pos_list_score - neg_list_score + 2, 0))

            # pos_list_h = tf.slice(pos_neg_list, [0, 0], [-1, 1])
            # pos_list_r = tf.slice(pos_neg_list, [0, 1], [-1, 1])
            # pos_list_t = tf.slice(pos_neg_list, [0, 2], [-1, 1])
            # neg_list_h = tf.slice(pos_neg_list, [0, 3], [-1, 1])
            # neg_list_r = tf.slice(pos_neg_list, [0, 4], [-1, 1])
            # neg_list_t = tf.slice(pos_neg_list, [0, 5], [-1, 1])

            # pos_list_h_embed = self.__l1_normalize(tf.nn.embedding_lookup(self.ent_embeddings, pos_list_h),1)
            # pos_list_r_embed = self.__l1_normalize(tf.nn.embedding_lookup(self.rel_embeddings, pos_list_r),1)
            # pos_list_t_embed = self.__l1_normalize(tf.nn.embedding_lookup(self.ent_embeddings, pos_list_t),1)
            # neg_list_h_embed = self.__l1_normalize(tf.nn.embedding_lookup(self.ent_embeddings, neg_list_h),1)
            # neg_list_r_embed = self.__l1_normalize(tf.nn.embedding_lookup(self.rel_embeddings, neg_list_r),1)
            # neg_list_t_embed = self.__l1_normalize(tf.nn.embedding_lookup(self.ent_embeddings, neg_list_t),1)

            # pos_list_score = (pos_list_h_embed + pos_list_r_embed - pos_list_t_embed) ** 2
            # neg_list_score = (neg_list_h_embed + neg_list_r_embed - neg_list_t_embed) ** 2

            # loss = tf.reduce_mean(tf.maximum(tf.reduce_sum(pos_list_score - neg_list_score + 2, 1),0))
            # self.ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings,dim=1) # L2 normalization of entity embeddings

            pos_list_h = tf.slice(pos_neg_list, [0, 0], [-1, 1])
            pos_list_r = tf.slice(pos_neg_list, [0, 1], [-1, 1])
            pos_list_t = tf.slice(pos_neg_list, [0, 2], [-1, 1])

            neg_list_1_h = tf.slice(pos_neg_list, [0, 3], [-1, 1])
            neg_list_1_r = tf.slice(pos_neg_list, [0, 4], [-1, 1])
            neg_list_1_t = tf.slice(pos_neg_list, [0, 5], [-1, 1])
            neg_list_2_h = tf.slice(pos_neg_list, [0, 6], [-1, 1])
            neg_list_2_r = tf.slice(pos_neg_list, [0, 7], [-1, 1])
            neg_list_2_t = tf.slice(pos_neg_list, [0, 8], [-1, 1])
            neg_list_3_h = tf.slice(pos_neg_list, [0, 9], [-1, 1])
            neg_list_3_r = tf.slice(pos_neg_list, [0, 10], [-1, 1])
            neg_list_3_t = tf.slice(pos_neg_list, [0, 11], [-1, 1])
            neg_list_4_h = tf.slice(pos_neg_list, [0, 12], [-1, 1])
            neg_list_4_r = tf.slice(pos_neg_list, [0, 13], [-1, 1])
            neg_list_4_t = tf.slice(pos_neg_list, [0, 14], [-1, 1])
            neg_list_5_h = tf.slice(pos_neg_list, [0, 15], [-1, 1])
            neg_list_5_r = tf.slice(pos_neg_list, [0, 16], [-1, 1])
            neg_list_5_t = tf.slice(pos_neg_list, [0, 17], [-1, 1])

            pos_list_h_embed = tf.nn.embedding_lookup(self.ent_embeddings, pos_list_h)
            pos_list_r_embed = tf.nn.embedding_lookup(self.rel_embeddings, pos_list_r)
            pos_list_t_embed = tf.nn.embedding_lookup(self.ent_embeddings, pos_list_t)

            neg_list_1_h_embed = tf.nn.embedding_lookup(self.ent_embeddings, neg_list_1_h)
            neg_list_1_r_embed = tf.nn.embedding_lookup(self.rel_embeddings, neg_list_1_r)
            neg_list_1_t_embed = tf.nn.embedding_lookup(self.ent_embeddings, neg_list_1_t)
            neg_list_2_h_embed = tf.nn.embedding_lookup(self.ent_embeddings, neg_list_2_h)
            neg_list_2_r_embed = tf.nn.embedding_lookup(self.rel_embeddings, neg_list_2_r)
            neg_list_2_t_embed = tf.nn.embedding_lookup(self.ent_embeddings, neg_list_2_t)
            neg_list_3_h_embed = tf.nn.embedding_lookup(self.ent_embeddings, neg_list_3_h)
            neg_list_3_r_embed = tf.nn.embedding_lookup(self.rel_embeddings, neg_list_3_r)
            neg_list_3_t_embed = tf.nn.embedding_lookup(self.ent_embeddings, neg_list_3_t)
            neg_list_4_h_embed = tf.nn.embedding_lookup(self.ent_embeddings, neg_list_4_h)
            neg_list_4_r_embed = tf.nn.embedding_lookup(self.rel_embeddings, neg_list_4_r)
            neg_list_4_t_embed = tf.nn.embedding_lookup(self.ent_embeddings, neg_list_4_t)
            neg_list_5_h_embed = tf.nn.embedding_lookup(self.ent_embeddings, neg_list_5_h)
            neg_list_5_r_embed = tf.nn.embedding_lookup(self.rel_embeddings, neg_list_5_r)
            neg_list_5_t_embed = tf.nn.embedding_lookup(self.ent_embeddings, neg_list_5_t)

            self.ent_embeddings = tf.scatter_update(self.ent_embeddings, pos_list_h, tf.nn.l2_normalize((pos_list_h_embed), dim=1))
            self.ent_embeddings = tf.scatter_update(self.ent_embeddings, pos_list_r, tf.nn.l2_normalize((pos_list_r_embed), dim=1))
            self.ent_embeddings = tf.scatter_update(self.ent_embeddings, pos_list_t, tf.nn.l2_normalize((pos_list_t_embed), dim=1))
            self.ent_embeddings = tf.scatter_update(self.ent_embeddings, neg_list_1_h, tf.nn.l2_normalize((neg_list_1_h_embed), dim=1))
            self.ent_embeddings = tf.scatter_update(self.ent_embeddings, neg_list_1_r, tf.nn.l2_normalize((neg_list_1_r_embed), dim=1))
            self.ent_embeddings = tf.scatter_update(self.ent_embeddings, neg_list_1_t, tf.nn.l2_normalize((neg_list_1_t_embed), dim=1))
            self.ent_embeddings = tf.scatter_update(self.ent_embeddings, neg_list_2_h, tf.nn.l2_normalize((neg_list_2_h_embed), dim=1))
            self.ent_embeddings = tf.scatter_update(self.ent_embeddings, neg_list_2_r, tf.nn.l2_normalize((neg_list_2_r_embed), dim=1))
            self.ent_embeddings = tf.scatter_update(self.ent_embeddings, neg_list_2_t, tf.nn.l2_normalize((neg_list_2_t_embed), dim=1))
            self.ent_embeddings = tf.scatter_update(self.ent_embeddings, neg_list_3_h, tf.nn.l2_normalize((neg_list_3_h_embed), dim=1))
            self.ent_embeddings = tf.scatter_update(self.ent_embeddings, neg_list_3_r, tf.nn.l2_normalize((neg_list_3_r_embed), dim=1))
            self.ent_embeddings = tf.scatter_update(self.ent_embeddings, neg_list_3_t, tf.nn.l2_normalize((neg_list_3_t_embed), dim=1))
            self.ent_embeddings = tf.scatter_update(self.ent_embeddings, neg_list_4_h, tf.nn.l2_normalize((neg_list_4_h_embed), dim=1))
            self.ent_embeddings = tf.scatter_update(self.ent_embeddings, neg_list_4_r, tf.nn.l2_normalize((neg_list_4_r_embed), dim=1))
            self.ent_embeddings = tf.scatter_update(self.ent_embeddings, neg_list_4_t, tf.nn.l2_normalize((neg_list_4_t_embed), dim=1))
            self.ent_embeddings = tf.scatter_update(self.ent_embeddings, neg_list_5_h, tf.nn.l2_normalize((neg_list_5_h_embed), dim=1))
            self.ent_embeddings = tf.scatter_update(self.ent_embeddings, neg_list_5_r, tf.nn.l2_normalize((neg_list_5_r_embed), dim=1))
            self.ent_embeddings = tf.scatter_update(self.ent_embeddings, neg_list_5_t, tf.nn.l2_normalize((neg_list_5_t_embed), dim=1))

            pos_list_score = tf.reduce_sum((pos_list_h_embed + pos_list_r_embed - pos_list_t_embed) ** 2,1)
            neg_list_1_score = tf.reduce_sum((neg_list_1_h_embed + neg_list_1_r_embed - neg_list_1_t_embed) ** 2,1)
            neg_list_2_score = tf.reduce_sum((neg_list_2_h_embed + neg_list_2_r_embed - neg_list_2_t_embed) ** 2,1)
            neg_list_3_score = tf.reduce_sum((neg_list_3_h_embed + neg_list_3_r_embed - neg_list_3_t_embed) ** 2,1)
            neg_list_4_score = tf.reduce_sum((neg_list_4_h_embed + neg_list_4_r_embed - neg_list_4_t_embed) ** 2,1)
            neg_list_5_score = tf.reduce_sum((neg_list_5_h_embed + neg_list_5_r_embed - neg_list_5_t_embed) ** 2,1)

            loss = tf.reduce_sum(tf.maximum(pos_list_score - neg_list_1_score + 2, 0)) + \
                    tf.reduce_sum(tf.maximum(pos_list_score - neg_list_2_score + 2, 0)) + \
                    tf.reduce_sum(tf.maximum(pos_list_score - neg_list_3_score + 2, 0)) + \
                    tf.reduce_sum(tf.maximum(pos_list_score - neg_list_4_score + 2, 0)) + \
                    tf.reduce_sum(tf.maximum(pos_list_score - neg_list_5_score + 2, 0))

            # with tf.device('/cpu'):
            #     loss = loss + regularizer_weight * tf.reduce_mean(tf.nn.l2_loss(self.ent_embeddings))

            return loss 

    def test(self, inputs, scope=None):
        with tf.variable_scope(scope or type(self).__name__) as scp:
            scp.reuse_variables()

            h = tf.nn.embedding_lookup(self.ent_embeddings, inputs[:, 0])
            t = tf.nn.embedding_lookup(self.ent_embeddings, inputs[:, 1])
            r = tf.nn.embedding_lookup(self.rel_embeddings, inputs[:, 2])

            # ent_mat = tf.transpose(self.ent_embeddings)

            # trh_res = tf.matmul(h+r, ent_mat)
            # _, tail_ids = tf.nn.top_k(trh_res, k=self.__n_entity)

            # hrh_res = tf.matmul(t-r, ent_mat)            
            # _, head_ids = tf.nn.top_k(hrh_res, k=self.__n_entity)

            t = AnnoyIndex(self.embed_dim)

            for i in xrange(self.n_entity):
                # v = tf.unpack(tf.slice(self.ent_embeddings, [i, 0], [1, -1]), axis=1)
                v = tf.unpack(tf.gather(self.ent_embeddings, [i]),axis=1)
                t.add_item(i, v)
                t.build(100)

            head_ids = list()
            tail_ids = list()

            for i in xrange(self.eval_batch):
                head_ids.append(t.get_nns_by_vector(tf.unpack(tf.slice(tf.add(h,r), [i, 0], [1, -1])), 10000))
                tail_ids.append(t.get_nns_by_vector(tf.unpack(tf.slice(tf.sub(t,r), [i, 0], [1, -1])), 10000))

            return head_ids, tail_ids


def train_ops(model, learning_rate=0.1, optimizer_str='gradient', regularizer_weight=1.0):
    # with tf.device('/cpu'):

    train_pos_neg_list = tf.placeholder(tf.int32, [None, 18])

    # model.ent_embeddings = tf.nn.l2_normalize(model.ent_embeddings,dim=1) # L2 normalization of entity embeddings

    loss = model.train(train_pos_neg_list,
                       regularizer_weight=regularizer_weight)

    if optimizer_str == 'gradient':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif optimizer_str == 'rms':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    elif optimizer_str == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    else:
        raise NotImplementedError("Does not support %s optimizer" % optimizer_str)

    grads = optimizer.compute_gradients(loss, tf.trainable_variables())

    op_train = optimizer.apply_gradients(grads)

    return train_pos_neg_list, loss, op_train

def embedding_ops(model):
    return model.ent_embeddings, model.rel_embeddings

def test_ops(model):
    # with tf.device('/cpu'):
    test_input = tf.placeholder(tf.int32, [None, 3])
    head_ids, tail_ids = model.test(test_input)

    return test_input, head_ids, tail_ids

def worker_func(dat, hr_t, tr_h):
    testing_data, head_pred, tail_pred = dat
    return test_evaluation(testing_data,head_pred,tail_pred,hr_t,tr_h)


def data_generator_func(train_data_list, tr_h, hr_t, n_entity, neg_sample, n_rel):
    dat = train_data_list
    # if dat is None:
    #     break
    pos_neg_list = list()

    htr = dat

    for idx in range(htr.shape[0]):
        if np.random.uniform(-1, 1) > 0:  # t r predict h
            assert htr[idx, 0] in tr_h[htr[idx, 1]][htr[idx, 2]]
            batch_ex = [htr[idx, 0], htr[idx, 2], htr[idx, 1]]
            # neg_h_list = random.sample([x for x in range(n_entity) if x not in tr_h[htr[idx, 1]][htr[idx, 2]]],neg_sample)
            neg_h_list = []
            while (len(neg_h_list) < neg_sample):
                s1 = random.sample(xrange(n_entity), (neg_sample - len(neg_h_list)))
                l1 = [x for x in s1 if x not in tr_h[htr[idx, 1]][htr[idx, 2]]]
                neg_h_list.extend(l1)

            for neg_head in neg_h_list:
                batch_ex.extend([neg_head, htr[idx, 2], htr[idx, 1]])
            pos_neg_list.append(batch_ex)
        else:  # h r predict t
            assert htr[idx, 1] in hr_t[htr[idx, 0]][htr[idx, 2]]
            batch_ex = [htr[idx, 0], htr[idx, 2], htr[idx, 1]]
            # neg_t_list = random.sample([x for x in range(n_entity) if x not in hr_t[htr[idx, 0]][htr[idx, 2]]],neg_sample)
            neg_t_list = []
            while (len(neg_t_list) < neg_sample):
                s1 = random.sample(xrange(n_entity), (neg_sample - len(neg_t_list)))
                l1 = [x for x in s1 if x not in hr_t[htr[idx, 0]][htr[idx, 2]]]
                neg_t_list.extend(l1)

            for neg_tail in neg_t_list:
                batch_ex.extend([htr[idx, 0], htr[idx, 2], neg_tail])
            pos_neg_list.append(batch_ex)          

    return (np.asarray(pos_neg_list, dtype=np.int32))

def test_evaluation(testing_data, head_pred, tail_pred, hr_t, tr_h):
    assert len(testing_data) == len(head_pred)
    assert len(testing_data) == len(tail_pred)

    mean_rank_h = list()
    mean_rank_t = list()
    filtered_mean_rank_h = list()
    filtered_mean_rank_t = list()

    for i in range(len(testing_data)):
        h = testing_data[i, 0]
        t = testing_data[i, 1]
        r = testing_data[i, 2]
        # mean rank

        mr = 0
        for val in head_pred[i]:
            if val == h:
                mean_rank_h.append(mr)
                break
            mr += 1

        mr = 0
        for val in tail_pred[i]:
            if val == t:
                mean_rank_t.append(mr)
            mr += 1

        # filtered mean rank
        fmr = 0
        for val in head_pred[i]:
            if val == h:
                filtered_mean_rank_h.append(fmr)
                break
            if t in tr_h and r in tr_h[t] and val in tr_h[t][r]:
                continue
            else:
                fmr += 1

        fmr = 0
        for val in tail_pred[i]:
            if val == t:
                filtered_mean_rank_t.append(fmr)
                break
            if h in hr_t and r in hr_t[h] and val in hr_t[h][r]:
                continue
            else:
                fmr += 1

    return (mean_rank_h, filtered_mean_rank_h), (mean_rank_t, filtered_mean_rank_t)


def main(_):
    parser = argparse.ArgumentParser(description='TransE.')
    parser.add_argument('--data', dest='data_dir', type=str, help="Data folder", default='./data/FB15k/')
    parser.add_argument('--lr', dest='lr', type=float, help="Learning rate", default=1e-2)
    parser.add_argument("--dim", dest='dim', type=int, help="Embedding dimension", default=256)
    parser.add_argument("--batch", dest='batch', type=int, help="Batch size", default=32)
    parser.add_argument("--worker", dest='n_worker', type=int, help="Evaluation worker", default=3)
    parser.add_argument("--generator", dest='n_generator', type=int, help="Data generator", default=10)
    parser.add_argument("--eval_batch", dest="eval_batch", type=int, help="Evaluation batch size", default=32)
    parser.add_argument("--save_dir", dest='save_dir', type=str, help="Model path", default='./transE')
    parser.add_argument("--load_model", dest='load_model', type=str, help="Model file", default="")
    parser.add_argument("--save_per", dest='save_per', type=int, help="Save per x iteration", default=1)
    parser.add_argument("--eval_per", dest='eval_per', type=int, help="Evaluate every x iteration", default=5)
    parser.add_argument("--max_iter", dest='max_iter', type=int, help="Max iteration", default=30)
    parser.add_argument("--summary_dir", dest='summary_dir', type=str, help="summary directory",
                        default='./transE_summary/')
    parser.add_argument("--keep", dest='drop_out', type=float, help="Keep prob (1.0 keep all, 0. drop all)",
                        default=0.5)
    parser.add_argument("--optimizer", dest='optimizer', type=str, help="Optimizer", default='gradient')
    parser.add_argument("--prefix", dest='prefix', type=str, help="model_prefix", default='DEFAULT')
    parser.add_argument("--loss_weight", dest='loss_weight', type=float, help="Weight on parameter loss", default=1e-2)
    parser.add_argument("--neg_weight", dest='neg_weight', type=float, help="Sampling weight on negative examples",
                        default=0.5)
    parser.add_argument("--save_per_batch",dest='save_per_batch',type=int,help='evaluate and save after every x batches',default=1000)
    parser.add_argument("--outfile_prefix",dest = 'outfile_prefix', type=str, help='The filename of output file is outfile_prefix.txt',default='test_output')
    parser.add_argument("--neg_sample",dest='neg_sample',type=int,help='No. of neg. samples per (h,r) or (t,r) pair',default=5)
    parser.add_argument("--fanout_thresh",dest='fanout_thresh',type=int,help='threshold on fanout of entities to be considered',default=2)
    parser.add_argument('--annoy_n_trees',dest='annoy_n_trees',type=int,help='builds a forest of n_trees trees',default=10)
    parser.add_argument('--annoy_search_k',dest='annoy_search_k',type=int,help='During the query it will inspect up to search_k nodes',default = -1)
    parser.add_argument('--eval_after',dest='eval_after',type=int,help='Evaluate after this many no. of epochs', default=4)

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print(args)

    model = TransE(args.data_dir, embed_dim=args.dim, fanout_thresh=args.fanout_thresh,eval_batch=args.eval_batch)

    train_pos_neg_list, \
    train_loss, train_op = train_ops(model, learning_rate=args.lr,
                                     optimizer_str=args.optimizer,
                                     regularizer_weight=args.loss_weight)

    get_embedding_op = embedding_ops(model)

    # test_input, test_head, test_tail = test_ops(model)
    f1 = open('%s/%s.txt' % (args.save_dir, args.outfile_prefix),'w')

    with tf.Session() as session:
        tf.global_variables_initializer().run()

        all_var = tf.all_variables()
        print 'printing all' , len(all_var),' TF variables:'
        for var in all_var:
            print var.name, var.get_shape()

        saver = tf.train.Saver(restore_sequentially=True)

        iter_offset = 0

        if args.load_model is not None and os.path.exists(args.load_model):
            saver.restore(session, args.load_model)
            iter_offset = int(args.load_model.split('.')[-2].split('_')[-1]) + 1
            f1.write("Load model from %s, iteration %d restored.\n" % (args.load_model, iter_offset))

        total_inst = model.n_train
        best_filtered_mean_rank = float("inf")

        f1.write("preparing training data...\n")
        nbatches_count = 0
        training_data_list = []
        # training_data_pos_neg_list = []

        for dat in model.raw_training_data(batch_size=args.batch):
            # raw_training_data_queue.put(dat)
            training_data_list.append(dat)
            # ps_list = data_generator_func(dat, model.tr_h, model.hr_t, model.n_entity, args.neg_sample, model.n_relation)
            # assert ps_list is not None
            # training_data_pos_neg_list.append(ps_list)
            nbatches_count += 1
        f1.write("training data prepared.\n")
        f1.write("No. of batches : %d\n" % nbatches_count)
        f1.close()

        start_time = timeit.default_timer()

        for n_iter in range(iter_offset, args.max_iter):            
            accu_loss = 0.
            ninst = 0
            # f1.close()

            for batch_id in range(nbatches_count):
                f1 = open('%s/%s.txt' % (args.save_dir, args.outfile_prefix),'a')

                pos_neg_list = data_generator_func(training_data_list[batch_id], model.tr_h, model.hr_t, model.n_entity, args.neg_sample, model.n_relation)
                #print data_e
                l, _ = session.run(
                    [train_loss, train_op], {train_pos_neg_list : pos_neg_list})

                accu_loss += l
                ninst += len(pos_neg_list) 

                # print('len(pos_neg_list) = %d\n' % len(pos_neg_list))

                if ninst % (5000) is not None:
                    f1.write(
                        '[%d sec](%d/%d) : %.2f -- loss : %.5f \n' % (
                            timeit.default_timer() - start_time, ninst, total_inst, float(ninst) / total_inst,
                            l))
                f1.close()

            f1 = open('%s/%s.txt' % (args.save_dir, args.outfile_prefix),'a')
            f1.write("")
            f1.write("iter %d avg loss %.5f, time %.3f\n" % (n_iter, accu_loss / ninst, timeit.default_timer() - start_time))

            save_path = saver.save(session,
                                        os.path.join(args.save_dir,
                                                     "TransE_" + str(args.prefix) + "_" + str(n_iter) + ".ckpt"))
            f1.write("Model saved at %s\n" % save_path)

            with tf.device('/cpu'):
                if n_iter > args.eval_after and (n_iter % args.eval_per == 0 or n_iter == args.max_iter - 1):

                    t = AnnoyIndex(model.embed_dim, metric='euclidean')

                    ent_embedding, rel_embedding = session.run(get_embedding_op,{train_pos_neg_list : pos_neg_list})
                    # sess = tf.InteractiveSession()
                    # with sess.as_default():
                    #     ent_embedding = model.ent_embeddings.eval()
                    print np.asarray(ent_embedding).shape
                    print np.asarray(rel_embedding).shape

                    # print ent_embedding[10,:]
                    # print rel_embedding[10,:]
                    print 'Index creation started'

                    for i in xrange(model.n_entity):
                        v = ent_embedding[i,:]
                        t.add_item(i, v)
                    t.build(args.annoy_n_trees)

                    print 'Index creation completed'

                    # n = int(0.0005 * model.n_entity)
                    n = 1000
                    # search_k = int(n * args.annoy_n_trees/100.0)
                    search_k = 1000

                    print 'No. of items = %d' % t.get_n_items()
                    print sum(t.get_item_vector(0))
                    print sum(ent_embedding[0,:])
                    assert sum(t.get_item_vector(0)) == sum(ent_embedding[0,:])

                    eval_dict = zip([model.raw_training_data], ['TRAIN'])

                    for data_func, test_type in eval_dict:
                        accu_mean_rank_h = list()
                        accu_mean_rank_t = list()
                        accu_filtered_mean_rank_h = list()
                        accu_filtered_mean_rank_t = list()

                        evaluation_count = 0
                        evaluation_batch = []
                        batch_id = 0
                        for testing_data in data_func(batch_size=args.eval_batch):
                            batch_id += 1
                            print 'test_type: %s, batch id: %d' % (test_type, batch_id) 
                            head_ids = list()
                            tail_ids = list()

                            for i in xrange(testing_data.shape[0]):
                                # try:
                                    # print (ent_embedding[testing_data[i,0],:] + rel_embedding[testing_data[i,2],:])
                                tail_ids.append(t.get_nns_by_vector((ent_embedding[testing_data[i,0],:] + rel_embedding[testing_data[i,2],:]), n,search_k))
                                head_ids.append(t.get_nns_by_vector((ent_embedding[testing_data[i,1],:] - rel_embedding[testing_data[i,2],:]), n,search_k))
                                # except:
                                #     print 'i = %d' % i
                                #     print 'testing_data[i,0] = %d' % testing_data[i,0]
                                #     print 'testing_data[i,1] = %d' % testing_data[i,1]
                                #     print 'testing_data[i,2] = %d' % testing_data[i,2]

                            # print head_ids
                            # print tail_ids
                            evaluation_batch.append((testing_data, head_ids, tail_ids))
                            evaluation_count += 1
                            if batch_id > 52662:
                                break

                        while evaluation_count > 0:
                            evaluation_count -= 1

                            # (mrh, fmrh), (mrt, fmrt) = result_queue.get()
                            (mrh, fmrh), (mrt, fmrt) = worker_func(evaluation_batch[evaluation_count-1],model.hr_t, model.tr_h)
                            accu_mean_rank_h += mrh
                            accu_mean_rank_t += mrt
                            accu_filtered_mean_rank_h += fmrh
                            accu_filtered_mean_rank_t += fmrt

                        f1.write(
                            "[%s] ITER %d [HEAD PREDICTION] MEAN RANK: %.1f FILTERED MEAN RANK %.1f HIT@10 %.3f FILTERED HIT@10 %.3f\n" %
                            (test_type, n_iter, np.mean(accu_mean_rank_h), np.mean(accu_filtered_mean_rank_h),
                             np.mean(np.asarray(accu_mean_rank_h, dtype=np.int32) < 10),
                             np.mean(np.asarray(accu_filtered_mean_rank_h, dtype=np.int32) < 10)))

                        f1.write(
                            "[%s] ITER %d [TAIL PREDICTION] MEAN RANK: %.1f FILTERED MEAN RANK %.1f HIT@10 %.3f FILTERED HIT@10 %.3f\n" %
                            (test_type, n_iter, np.mean(accu_mean_rank_t), np.mean(accu_filtered_mean_rank_t),
                             np.mean(np.asarray(accu_mean_rank_t, dtype=np.int32) < 10),
                             np.mean(np.asarray(accu_filtered_mean_rank_t, dtype=np.int32) < 10)))

            f1.close()

if __name__ == '__main__':
    tf.app.run()
