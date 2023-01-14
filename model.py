from Recommender import GraphRecommender, DMIRecommender
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from scipy.sparse import coo_matrix
import numpy as np
from util.config import OptionConf,bpr_loss
import os
from util import config
from math import sqrt
from DFload import DFload
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import csv
from time import strftime,localtime,time

class model(GraphRecommender, DMIRecommender): #继承类DDIRecommender,GraphRecommender
    def __init__(self, conf, trainingSet=None, testSet=None, fold='[1]'):
        self.config = config
        GraphRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, fold=fold)
        DMIRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, fold=fold)

    def readConfiguration(self):
        super(model, self).readConfiguration()
        args = config.OptionConf(self.config['model'])
        self.n_layers = int(args['-n_layer'])
        self.ss_rate = float(args['-ss_rate'])  

    #针对DDI,DTI,DDiI,DSI多个任务的训练数据的输入,不管任务是谁，都是分子
    def buildSparseTraMatrix(self): #train data
        row, col, entries = [], [], []
        if self.config['Task.name'] == 'DrugDrug':
            print('DrugDrug_train')
            for pair in self.data.trainingData:
                row += [self.data.drug[pair[0]]]
                row += [self.data.molecular[pair[1]]]
                entries += [1.0]
                col += [self.data.drug[pair[1]]] #self.data.drug是存储药物的list，self.data.drug[pair[0]]是得到药物在drug list的idx
                col += [self.data.molecular[pair[0]]]
                entries += [1.0]
        else:
            for pair in self.data.trainingData:
            # symmetric matrix
                row += [self.data.drug[pair[0]]]
                col += [self.data.molecular[pair[1]]]
                entries += [1.0]
        TraMatrix = coo_matrix((entries, (row, col)), shape=(self.num_drugs,self.num_molecular),dtype=np.float32)
        #print('TraMatrix', TraMatrix.shape)
        return TraMatrix 
    
    def buildJointAdjacency(self):
        if self.config['Task.name'] == 'DrugDrug':
            indices = [[self.data.drug[item[0]], self.data.molecular[item[1]]] for item in self.data.trainingData]
            indices += [[self.data.drug[item[0]], self.data.molecular[item[1]]] for item in self.data.trainingData]
            values = [float(item[2]) / sqrt(len(self.data.trainSet_u[item[0]])) / sqrt(len(self.data.trainSet_i[item[1]])) for item in self.data.trainingData]
            values += [float(item[2]) / sqrt(len(self.data.trainSet_u[item[0]])) / sqrt(len(self.data.trainSet_i[item[1]])) for item in self.data.trainingData]
            norm_adj = tf.SparseTensor(indices=indices, values=values,
                                dense_shape=[self.num_drugs, self.num_molecular])            
        else:    
            indices = [[self.data.drug[item[0]], self.data.molecular[item[1]]] for item in self.data.trainingData]
            values = [float(item[2]) / sqrt(len(self.data.trainSet_u[item[0]])) / sqrt(len(self.data.trainSet_i[item[1]])) for item in self.data.trainingData]
            norm_adj = tf.SparseTensor(indices=indices, values=values,
                                dense_shape=[self.num_drugs, self.num_molecular])
        return norm_adj
    
    def buildSparseDDMatrix(self): #d-d无向，关联数据存为对称矩阵
        row, col, entries = [], [], []
        for pair in self.data.DDIrelation:
            if pair[0] in self.data.drug and pair[1] in self.data.drug:
                # symmetric matrix
                row += [self.data.drug[pair[0]]]
                row += [self.data.drug[pair[1]]]
                entries += [1.0]
                col += [self.data.drug[pair[1]]] #self.data.drug是存储药物的list，self.data.drug[pair[0]]是得到药物在drug list的idx
                col += [self.data.drug[pair[0]]]
                entries += [1.0]
        DDIMatrix = coo_matrix((entries, (row, col)), shape=(self.num_drugs,self.num_drugs),dtype=np.float32) #row行和col列存了1，其余位置皆是0
        #print('DDIMatrix', DDIMatrix.shape)
        return DDIMatrix

    def buildSparseDFMatrix(self): #d-f
        row, col, entries = [], [], []
        for pair in self.data.DFIrelation:
            if pair[0] in self.data.drug:
            # symmetric matrix
                row += [self.data.drug[pair[0]]]
                col += [self.data.fragment[pair[1]]]
                entries += [1.0]
        DFIMatrix = coo_matrix((entries, (row, col)), shape=(self.num_drugs,self.num_fragments),dtype=np.float32)
        #print('DFIMatrix', DFIMatrix.shape)
        return DFIMatrix

    def buildSparseFFMatrix(self): #f-f
        row, col, entries = [], [], []
        for pair in self.data.FFIrelation:
            # symmetric matrix
            row += [self.data.fragment[pair[0]]]
            col += [self.data.fragment[pair[1]]]
            entries += [1.0]
        FFIMatrix = coo_matrix((entries, (row, col)), shape=(self.num_fragments,self.num_fragments),dtype=np.float32)
        #print('FFIMatrix', FFIMatrix.shape)
        return FFIMatrix

    def buildSparseDTMatrix(self): #d-t
        row, col, entries = [], [], []
        for pair in self.data.DTIrelation:
            if pair[0] in self.data.drug:
            # symmetric matrix
                row += [self.data.drug[pair[0]]]
                col += [self.data.target[pair[1]]]
                entries += [1.0]
        DTIMatrix = coo_matrix((entries, (row, col)), shape=(self.num_drugs,self.num_targets),dtype=np.float32)
        #print('DTIMatrix', DTIMatrix.shape)
        return DTIMatrix

    def buildSparseDDiMatrix(self): #d-di
        row, col, entries = [], [], []
        for pair in self.data.DDiIrelation:
            if pair[0] in self.data.disease:
            # symmetric matrix
                row += [self.data.drug[pair[0]]]
                col += [self.data.disease[pair[1]]]
                entries += [1.0]
        DDiIMatrix = coo_matrix((entries, (row, col)), shape=(self.num_drugs,self.num_diseases),dtype=np.float32)
        #print('DDiIMatrix', DDiIMatrix.shape)
        return DDiIMatrix

    def buildSparseDSMatrix(self): #d-s
        row, col, entries = [], [], []
        for pair in self.data.DSIrelation:
            if pair[0] in self.data.drug:
            # symmetric matrix
                row += [self.data.drug[pair[0]]]
                col += [self.data.sideeffect[pair[1]]]
                entries += [1.0]
        DSIMatrix = coo_matrix((entries, (row, col)), shape=(self.num_drugs,self.num_sideeffect),dtype=np.float32)
        #print('DSIMatrix', DSIMatrix.shape)
        return DSIMatrix
 
    def buildMotifInducedAdjacencyMatrix(self): #related-motif to bulid hypergraphs
        #print('OptionConf',self.config['Task.name'])
        if self.config['Task.name'] == 'DrugDrug': #d-d
            print('d-d')
            S = self.buildSparseTraMatrix()
        else:
            S = self.buildSparseDDMatrix()
            
        if self.config['Task.name'] == 'DrugTarget': #d-t
            print('d-t')
            W = self.buildSparseTraMatrix()
        else:
            W = self.buildSparseDTMatrix() #d-t 

        if self.config['Task.name'] == 'DrugDisease': #d-di
            print('d-di')
            G = self.buildSparseTraMatrix()
        else:
            G = self.buildSparseDDiMatrix()
            
        if self.config['Task.name'] == 'DrugSideeffect': #d-s
            print('d-s')
            V = self.buildSparseTraMatrix()
        else:
            V = self.buildSparseDSMatrix()
            
        Y = self.buildSparseDFMatrix() #d-f
        Z = self.buildSparseFFMatrix() #f-f
        F = self.buildSparseTraMatrix() #train data
        Ft = F.T
        self.drugAdjacency = Y.tocsr()
        self.fragmentAdjacency = Y.T.tocsr()

        A1 = (S.dot(S)).multiply(S)
        A2 = (Y.dot(Y.T)).multiply(S)
        C1 = ((Y.dot(Z)).dot(Y.T)).multiply(S)
        A3 = C1 + C1.T
        A4 = S.dot(S) - A1
        A5 = Y.dot(Y.T) - A2
        C2 = (Y.dot(Z)).dot(Y.T)
        A6 = C2 + C2.T - A3

        I_j = sum(([A1,A2,A3]))
        I_j = I_j.multiply(1.0/I_j.sum(axis=1).reshape(-1, 1)) ##addition and row-normalization

        I_p = sum([A4,A5,A6])
        I_p = I_p.multiply(I_p>1)
        I_p = I_p.multiply(1.0/I_p.sum(axis=1).reshape(-1, 1))
        
        A7 = (W.dot(W.T)).multiply(S)
        A8 = (G.dot(G.T)).multiply(S)
        A9 = (V.dot(V.T)).multiply(S)                         
        A10 = W.dot(W.T)-A7 #A12
        A11 = G.dot(G.T)-A8 #A13
        A12 = V.dot(V.T)-A9 #A14
        
        I_i = sum([A7,A8,A9])
        I_i = I_i.multiply(1.0/I_i.sum(axis=1).reshape(-1, 1)) ##addition and row-normalization
        I_u = sum([A10,A11,A12])
        I_u = I_u.multiply(I_u>1)
        I_u = I_u.multiply(1.0/I_u.sum(axis=1).reshape(-1, 1))
        
        return [I_j, I_p, I_i, I_u]
    
    def adj_to_sparse_tensor(self,adj):
        adj = adj.tocoo()
        indices = np.mat(list(zip(adj.row, adj.col)))
        adj = tf.SparseTensor(indices, adj.data.astype(np.float32), adj.shape)
        return adj

    def initModel(self):
        super(model, self).initModel() #super() 函数是用于调用父类(超类)的一个方法
        M_matrices = self.buildMotifInducedAdjacencyMatrix()
        self.weights = {}
        initializer = tf.keras.initializers.glorot_normal()
        self.n_branch = 5
        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")

        #define learnable paramters
        for i in range(self.n_branch):
            self.weights['gating%d' % (i+1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='g_W_%d_1' % (i + 1))
            self.weights['gating_bias%d' %(i+1)] = tf.Variable(initializer([1, self.emb_size]), name='g_W_b_%d_1' % (i + 1))
            self.weights['sgating%d' % (i + 1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='sg_W_%d_1' % (i + 1))
            self.weights['sgating_bias%d' % (i + 1)] = tf.Variable(initializer([1, self.emb_size]), name='sg_W_b_%d_1' % (i + 1))
        self.weights['attention'] = tf.Variable(initializer([1, self.emb_size]), name='at')
        self.weights['attention_mat'] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='atm')

        #define messege passing layer
        for i in range(self.n_branch+1):
            self.weights['mpassing%d' % (i+1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='mp_W_%d_1' % (i + 1))
            self.weights['mpassing_bias%d' %(i+1)] = tf.Variable(initializer([1, self.emb_size]), name='mp_W_b_%d_1' % (i + 1))
        
        #define inline functions
        def self_gating(em,branch):
            return tf.multiply(em,tf.nn.sigmoid(tf.matmul(em,self.weights['gating%d' % branch])+self.weights['gating_bias%d' %branch]))
        def messege_passing(em,branch):
            return tf.nn.relu(tf.matmul(em,self.weights['mpassing%d' % branch])+self.weights['mpassing_bias%d' %branch])
        def self_supervised_gating(em, branch):
            return tf.multiply(em,tf.nn.sigmoid(tf.matmul(em, self.weights['sgating%d' % branch])+self.weights['sgating_bias%d' % branch]))
        def branch_attention(*branch_embeddings):
            weights = []
            for embedding in branch_embeddings:
                weights.append(tf.reduce_sum(tf.multiply(self.weights['attention'], tf.matmul(embedding, self.weights['attention_mat'])),1))
            score = tf.nn.softmax(tf.transpose(weights))
            mixed_embeddings = 0
            for i in range(len(weights)):
                mixed_embeddings += tf.transpose(tf.multiply(tf.transpose(score)[i], tf.transpose(branch_embeddings[i])))
            return mixed_embeddings,score
        #initialize adjacency matrices
        I_j = M_matrices[0]
        I_j = self.adj_to_sparse_tensor(I_j)
        I_p = M_matrices[1]
        I_p = self.adj_to_sparse_tensor(I_p)
        I_i = M_matrices[2]
        I_i = self.adj_to_sparse_tensor(I_i)
        I_u = M_matrices[3]
        I_u = self.adj_to_sparse_tensor(I_u)
        R = self.buildJointAdjacency()
        #self-gating
        drug_embeddings_c1 = self_gating(self.drug_embeddings,1)
        drug_embeddings_c2 = self_gating(self.drug_embeddings, 2)
        drug_embeddings_c3 = self_gating(self.drug_embeddings,3)
        drug_embeddings_c4 = self_gating(self.drug_embeddings, 4)
        
        all_embeddings_c1 = [drug_embeddings_c1]
        all_embeddings_c2 = [drug_embeddings_c2]
        all_embeddings_c3 = [drug_embeddings_c3]
        all_embeddings_c4 = [drug_embeddings_c4]

        simple_drug_embeddings = self_gating(self.drug_embeddings,5)
        all_embeddings_simple_drug = [simple_drug_embeddings]

        molecular_embeddings = self.molecular_embeddings
        all_embeddings_i = [molecular_embeddings]

        self.ss_loss = 0
        #multi-branches graph attention 
        for k in range(self.n_layers):
            mixed_embedding_drug = branch_attention(drug_embeddings_c1, drug_embeddings_c2, drug_embeddings_c3, drug_embeddings_c4)[0] + simple_drug_embeddings / 2#用attention而不是sum

            #branch j
            temp = tf.matmul(drug_embeddings_c1,tf.transpose(drug_embeddings_c1)) 
            I_j_temp = tf.sparse_softmax(I_j.__mul__(temp))
            drug_embeddings_c1 = tf.sparse_tensor_dense_matmul(I_j_temp,drug_embeddings_c1)
            
            drug_embeddings_c1 = messege_passing(drug_embeddings_c1,1)  #messege passing layer
            norm_embeddings = tf.math.l2_normalize(drug_embeddings_c1, axis=1)
            all_embeddings_c1 += [norm_embeddings]

            #branch p
            temp = tf.matmul(drug_embeddings_c2,tf.transpose(drug_embeddings_c2)) 
            I_p_temp = tf.sparse_softmax(I_p.__mul__(temp))            
            drug_embeddings_c2 = tf.sparse_tensor_dense_matmul(I_p_temp, drug_embeddings_c2)

            drug_embeddings_c2 = messege_passing(drug_embeddings_c2,2)
            norm_embeddings = tf.math.l2_normalize(drug_embeddings_c2, axis=1)
            all_embeddings_c2 += [norm_embeddings]
            
            #branch i
            temp = tf.matmul(drug_embeddings_c3,tf.transpose(drug_embeddings_c3)) 
            I_i_temp = tf.sparse_softmax(I_i.__mul__(temp))             
            drug_embeddings_c3 = tf.sparse_tensor_dense_matmul(I_i_temp,drug_embeddings_c3)

            drug_embeddings_c3 = messege_passing(drug_embeddings_c3,3)
            norm_embeddings = tf.math.l2_normalize(drug_embeddings_c3, axis=1)
            all_embeddings_c3 += [norm_embeddings]

            #branch u
            temp = tf.matmul(drug_embeddings_c4,tf.transpose(drug_embeddings_c4)) 
            I_u_temp = tf.sparse_softmax(I_u.__mul__(temp))            
            drug_embeddings_c4 = tf.sparse_tensor_dense_matmul(I_u_temp, drug_embeddings_c4)

            drug_embeddings_c4 = messege_passing(drug_embeddings_c4,4)
            norm_embeddings = tf.math.l2_normalize(drug_embeddings_c4, axis=1)
            all_embeddings_c4 += [norm_embeddings]

            # drug_related molecular convolution
            new_molecular_embeddings = tf.sparse_tensor_dense_matmul(tf.sparse.transpose(R), mixed_embedding_drug)
            #new_molecular_embeddings = messege_passing(new_molecular_embeddings,5) #DTI, DDiI task no have the line
            norm_embeddings = tf.math.l2_normalize(new_molecular_embeddings, axis=1)
            all_embeddings_i += [norm_embeddings]
            
            simple_drug_embeddings = tf.sparse_tensor_dense_matmul(R, molecular_embeddings)
            #simple_drug_embeddings = messege_passing(simple_drug_embeddings,5) #DTI, DDiI task no have the line
            all_embeddings_simple_drug += [tf.math.l2_normalize(simple_drug_embeddings, axis=1)] 
            molecular_embeddings = new_molecular_embeddings
            
        #averaging the branch-specific drug embeddings
        drug_embeddings_c1 = tf.reduce_sum(all_embeddings_c1, axis=0)
        drug_embeddings_c2 = tf.reduce_sum(all_embeddings_c2, axis=0)
        drug_embeddings_c3 = tf.reduce_sum(all_embeddings_c3, axis=0)
        drug_embeddings_c4 = tf.reduce_sum(all_embeddings_c4, axis=0)
        simple_drug_embeddings = tf.reduce_sum(all_embeddings_simple_drug, axis=0)
        molecular_embeddings = tf.reduce_sum(all_embeddings_i, axis=0)
        self.final_molecular_embeddings = molecular_embeddings

        #aggregating branch-specific embeddings
        self.final_drug_embeddings,self.attention_score = branch_attention(drug_embeddings_c1,drug_embeddings_c2, drug_embeddings_c3,drug_embeddings_c4)
        self.final_drug_embeddings += simple_drug_embeddings/2


        #create self-supervised loss
        self.ss_loss += self.hierarchical_self_supervision(self_supervised_gating(self.final_drug_embeddings,1), I_j)
        self.ss_loss += self.hierarchical_self_supervision(self_supervised_gating(self.final_drug_embeddings,2), I_p)
        self.ss_loss += self.hierarchical_self_supervision(self_supervised_gating(self.final_drug_embeddings,3), I_i)
        self.ss_loss += self.hierarchical_self_supervision(self_supervised_gating(self.final_drug_embeddings,4), I_u)        

        #embedding look-up
        self.batch_neg_molecular_emb = tf.nn.embedding_lookup(self.final_molecular_embeddings, self.neg_idx) #d-d中对于drug相关的drug_neg_idx在已知的drug-drug中未出现
        self.batch_drug_emb = tf.nn.embedding_lookup(self.final_drug_embeddings, self.u_idx)
        self.batch_pos_molecular_emb = tf.nn.embedding_lookup(self.final_molecular_embeddings, self.v_idx) #d-d中对于drug相关的drug_v_idx在已知的drug-drug中出现

    def hierarchical_self_supervision(self,em,adj):
        def row_shuffle(embedding):
            return tf.gather(embedding, tf.random.shuffle(tf.range(tf.shape(embedding)[0])))
        def row_column_shuffle(embedding):
            corrupted_embedding = tf.transpose(tf.gather(tf.transpose(embedding), tf.random.shuffle(tf.range(tf.shape(tf.transpose(embedding))[0]))))
            corrupted_embedding = tf.gather(corrupted_embedding, tf.random.shuffle(tf.range(tf.shape(corrupted_embedding)[0])))
            return corrupted_embedding
        def score(x1,x2):
            return tf.reduce_sum(tf.multiply(x1,x2),1)
        drug_embeddings = em
        #drug_embeddings = tf.math.l2_normalize(em,1) #For Douban, normalization is needed.
        edge_embeddings = tf.sparse_tensor_dense_matmul(adj,drug_embeddings)
        #Local MIM
        pos = score(drug_embeddings,edge_embeddings)
        neg1 = score(row_shuffle(drug_embeddings),edge_embeddings)
        neg2 = score(row_column_shuffle(edge_embeddings),drug_embeddings)
        local_loss = tf.reduce_sum(-tf.log(tf.sigmoid(pos-neg1))-tf.log(tf.sigmoid(neg1-neg2)))
        #Global MIM
        graph = tf.reduce_mean(edge_embeddings,0)
        pos = score(edge_embeddings,graph)
        neg1 = score(row_column_shuffle(edge_embeddings),graph)
        global_loss = tf.reduce_sum(-tf.log(tf.sigmoid(pos-neg1)))
        return global_loss+local_loss

    def trainModel(self):
        rec_loss = bpr_loss(self.batch_drug_emb, self.batch_pos_molecular_emb, self.batch_neg_molecular_emb)
        reg_loss = 0
        for key in self.weights:
            reg_loss += 0.001*tf.nn.l2_loss(self.weights[key])
        reg_loss += self.regU * (tf.nn.l2_loss(self.drug_embeddings) + tf.nn.l2_loss(self.molecular_embeddings))
        total_loss = rec_loss+reg_loss + self.ss_rate*self.ss_loss
        opt = tf.train.AdamOptimizer(self.lRate)
        train_op = opt.minimize(total_loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        # Suggested Maximum epoch Setting: LastFM 120 Douban 30 Yelp 30
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(self.next_batch_pairwise()):
                drug_idx, i_idx, j_idx = batch
                _, l1 = self.sess.run([train_op, rec_loss],
                                     feed_dict={self.u_idx: drug_idx, self.neg_idx: j_idx, self.v_idx: i_idx})
                print(self.foldInfo,'training:', epoch + 1, 'batch', n, 'rec loss:', l1)#,'ss_loss',l2
            self.U, self.V = self.sess.run([self.final_drug_embeddings, self.final_molecular_embeddings])
            self.ranking_performance(epoch)
        self.U,self.V = self.bestU,self.bestV

        drug_feature_dict = {self.data.id2drug[i]:[list(self.U[i])] for i in range(self.num_drugs)}
        mol_feature_dict = {self.data.id2molecular[i]:[list(self.V[i])] for i in range(self.num_molecular)}

        currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        outDir = self.output['-dir']
        drug_fileName = self.config['model.name'] + '@' + currentTime + '-Drug-feature' + self.foldInfo + '.csv'
        mol_fileName = self.config['model.name'] + '@' + currentTime + '-Mol-feature' + self.foldInfo + '.csv'

        with open(outDir+drug_fileName, 'w', newline='') as csvfile:
            dictwriter = csv.writer(csvfile)        

            # Define and write the header row with enough 'listX' columns
            header = ['Drug', 'feature'] 
            dictwriter.writerow(header)

            # Iterate through each entry in the dictionary, writing each row
            for key, value in drug_feature_dict.items():
                # Extend the list with blank values (not totally necessary, but keeps the csv file uniform)
                row = [key] + value 
                dictwriter.writerow(row)
        
        with open(outDir+mol_fileName, 'w', newline='') as csvfile:
            dictwriter = csv.writer(csvfile)        

            # Define and write the header row with enough 'listX' columns
            header = ['Mol', 'feature'] 
            dictwriter.writerow(header)

            # Iterate through each entry in the dictionary, writing each row
            for key, value in mol_feature_dict.items():
                # Extend the list with blank values (not totally necessary, but keeps the csv file uniform)
                row = [key] + value 
                dictwriter.writerow(row)
    def saveModel(self):
        self.bestU, self.bestV= self.sess.run([self.final_drug_embeddings, self.final_molecular_embeddings,])

    def __globalAverage(self):
        total = sum(self.drugMeans.values())
        if total==0:
            self.globalMean = 0
        else:
            self.globalMean = total/len(self.drugMeans)
               
    def predictForRanking(self, u):
        'invoked to rank all the fragments for the drug'
        if self.data.containsdrug(u):
            u = self.data.getdrugId(u)
            S1 = self.V.dot(self.U[u])
            #S2 = np.array(S1)
            #print(type(S1))
            return S1
            #return self.U.dot(self.U[u])预测D-D？
        else:
            return [self.data.globalMean] * self.num_molecular
