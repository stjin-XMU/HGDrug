import numpy as np
import scipy.sparse as sp
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from random import shuffle,randint,choice
from util import config
from DFload import DFload
from os.path import abspath
from time import strftime,localtime,time
from util.config import OptionConf,Log,find_k_largest,Measure,FileIO

class Recommender(object):
    def __init__(self,conf,trainingSet,testSet, fold='[1]'):
        self.config = conf
        self.data = None
        self.isSaveModel = False
        self.ranking = None
        self.isLoadModel = False
        self.output = None
        self.isOutput = True
        self.data = DFload(self.config, trainingSet, testSet)
        self.foldInfo = fold
        self.evalSettings = OptionConf(self.config['evaluation.setup'])
        self.measure = []
        self.recOutput = []
        self.train_num_drugs, self.train_num_molecular, self.train_size = self.data.trainingSize()
        self.num_fragments = self.data.FragmentSize()
        self.num_drugs = self.data.DrugSize()
        self.num_molecular = self.data.molecularSize()
        self.num_targets = self.data.TargetSize()
        self.num_diseases = self.data.DiseaseSize()
        self.num_sideeffect = self.data.sideeffectSize()

    def initializing_log(self):
        currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        self.log = Log(self.modelName, self.modelName + self.foldInfo + ' ' + currentTime)
        #save configuration
        self.log.add('### model configuration ###')
        for k in self.config.config:
            self.log.add(k+'='+self.config[k])

    def readConfiguration(self):
        self.modelName = self.config['model.name']
        self.output = OptionConf(self.config['output.setup'])
        self.isOutput = self.output.isMainOn()
        self.ranking = OptionConf(self.config['drug.ranking'])

    def printAlgorConfig(self):
        "show model's configuration"
        print('Model:',self.config['model.name'])
        print('Train dataset:',abspath(self.config['Task']))
        if OptionConf(self.config['evaluation.setup']).contains('-testSet'):
            print('Test set:', abspath(OptionConf(self.config['evaluation.setup'])['-testSet']))
        #print dataset statistics
        print('All Drug and molecular set size: (drug count: %d, molecular count: %d)' %(self.num_drugs, self.num_molecular, ))
        print('Training set size: (train_drug count: %d, train_molecular count: %d, train record count: %d)' %(self.data.trainingSize()))
        print('Test set size: (test_drug count: %d, test_molecular count %d, train record count: %d)' %(self.data.testSize()))
        print('='*80)
        #print specific parameters if applicable
        if self.config.contains(self.config['model.name']):
            parStr = ''
            args = OptionConf(self.config[self.config['model.name']])
            for key in args.keys():
                parStr+=key[1:]+':'+args[key]+'  '
            print('Specific parameters:',parStr)
            print('=' * 80)

    def initModel(self):
        pass

    def trainModel(self):
        'build the model (for model-based Models )'
        pass

    def trainModel_tf(self):
        'training model on tensorflow'
        pass

    def saveModel(self):
        pass

    def loadModel(self):
        pass

    #for rating prediction
    def predictForRating(self, u, i):
        pass

    #for fragment prediction
    def predictForRanking(self,u):
        pass

    def checkRatingBoundary(self,prediction):
        if prediction > self.data.rScale[-1]:
            return self.data.rScale[-1]
        elif prediction < self.data.rScale[0]:
            return self.data.rScale[0]
        else:
            return round(prediction,3)

    def evalDMI(self):
        res = list() #used to contain the text of the result
        res.append('drugId  molecularId  original  prediction\n')
        #predict
        for ind,entry in enumerate(self.data.testData):
            drug,molecular,rating = entry
            #predict
            prediction = self.predictForRating(drug, molecular)
            #denormalize
            #prediction = denormalize(prediction,self.data.rScale[-1],self.data.rScale[0])
            #####################################
            pred = self.checkRatingBoundary(prediction)
            # add prediction in order to measure
            self.data.testData[ind].append(pred)
            res.append(drug+' '+ molecular +' '+str(rating)+' '+str(pred)+'\n')
        currentTime = strftime("%Y-%m-%d %H-%M-%S",localtime(time()))
        #output prediction result
        if self.isOutput:
            outDir = self.output['-dir']
            fileName = self.config['model.name']+'@'+currentTime+ self.config['task']+'-predictions'+self.foldInfo+'.txt'
            FileIO.writeFile(outDir,fileName,res)
            print('The result has been output to ',abspath(outDir),'.')
        #output evaluation result
        outDir = self.output['-dir']
        fileName = self.config['model.name'] + '@'+currentTime +'-measure'+ self.foldInfo + '.txt'
        self.measure = Measure.ratingMeasure(self.data.testData)
        FileIO.writeFile(outDir, fileName, self.measure)
        self.log.add('###Evaluation Results###')
        self.log.add(self.measure)
        print('The result of %s %s:\n%s' % (self.modelName, self.foldInfo, ''.join(self.measure)))

    def evalRanking(self):
        if self.ranking.contains('-topN'):
            top = self.ranking['-topN'].split(',')
            top = [int(num) for num in top]
            N = max(top)
            if N > 1000 or N < 1:
                print('N can not be larger than 100! It has been reassigned to 10')
                N = 10
        else:
            print('No correct evaluation metric is specified!')
            exit(-1)
        self.recOutput.append('drugId: recommendations in (molecularId, ranking score) pairs, * means the drug matches.\n')
        # predict
        recList = {}
        drugCount = len(self.data.testSet_u)
        #print('drugCount:', drugCount )
        rawRes = {}#20220411
        topN_rawRes = {}
        for i, drug in enumerate(self.data.testSet_u): #输出结果评估得是在test里面包括得药物得topN，而非全部药物。
            line = drug + ':'
            candidates = self.predictForRanking(drug)
            # predictedfragments = denormalize(predictedfragments, self.data.rScale[-1], self.data.rScale[0])
            ratedList, ratingList = self.data.drugRated(drug) #

            for molecular in ratedList:
                candidates[self.data.molecular[molecular]] = 0

            ids,scores = find_k_largest(N,candidates)
            molecular_names = [self.data.id2molecular[iid] for iid in ids]
            recList[drug] = list(zip(molecular_names,scores))
            topN_rawRes[drug] = dict(zip(molecular_names,scores))#20220411

            #20220412
            all_candidates = []
            for all_id, all_score in enumerate(candidates[:]):
                all_candidates.append((all_id, all_score))
            #all_candidates.sort(key=lambda d: d[1], reverse=True) #目前算AUC的方法不用排序

            all_ids = [molecular[0] for molecular in all_candidates]
            all_molecular_names = [self.data.id2molecular[iid] for iid in all_ids]
            all_scores = [molecular[1] for molecular in all_candidates]
            rawRes[drug] = dict(zip(all_molecular_names, all_scores))

            if i % 100 == 0:
                print(self.modelName, self.foldInfo, 'progress:' + str(i) + '/' + str(drugCount))
            for molecular in recList[drug]:
                line += ' (' + molecular[0] + ',' + str(molecular[1]) + ')'
                if molecular[0] in self.data.testSet_u[drug]:
                    line += '*'
            line += '\n'
            self.recOutput.append(line)
        #print('rawRes:', rawRes)
        currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        # output prediction result
        if self.isOutput:
            outDir = self.output['-dir']
            fileName = self.config['model.name'] + '@' + currentTime + '-top-' + str(
            N) + 'drugs' + self.foldInfo + '.txt'
            FileIO.writeFile(outDir, fileName, self.recOutput)
            print('The result has been output to ', abspath(outDir), '.')
        # output evaluation result
        if self.evalSettings.contains('-predict'):
            #no evalutation
            exit(0)
        outDir = self.output['-dir']
        fileName = self.config['model.name'] + '@' + currentTime + '-measure' + self.foldInfo + '.txt'
        self.measure = Measure.rankingMeasure(self.data.testSet_u, self.data.trainSet_u, recList, topN_rawRes, rawRes, top)#20220411
        self.log.add('###Evaluation Results###')
        self.log.add(self.measure)
        FileIO.writeFile(outDir, fileName, self.measure)
        print('The result of %s %s:\n%s' % (self.modelName, self.foldInfo, ''.join(self.measure)))

    def execute(self):
        self.readConfiguration()
        self.initializing_log()
        if self.foldInfo == '[1]':
            self.printAlgorConfig()
        #load model from disk or build model
        if self.isLoadModel:
            print('Loading model %s...' %self.foldInfo)
            self.loadModel()
        else:
            print('Initializing model %s...' %self.foldInfo)
            self.initModel()
            print('Building Model %s...' %self.foldInfo)
            try:
                if self.evalSettings.contains('-tf'):
                    import tensorflow
                    self.trainModel_tf()
                else:
                    self.trainModel()
            except ImportError:
                self.trainModel()
        #rating prediction or fragment ranking
        print('Predicting %s...' %self.foldInfo)
        if self.ranking.isMainOn():
            self.evalRanking()
        else:
            self.evalDMI()
        #save model
        if self.isSaveModel:
            print('Saving model %s...' %self.foldInfo)
            self.saveModel()
        return self.measure

class IterativeRecommender(Recommender):
    def __init__(self,conf,trainingSet,testSet, fold='[1]'):
        super(IterativeRecommender, self).__init__(conf,trainingSet,testSet, fold)
        self.bestPerformance = []
        self.earlyStop = 0

    def readConfiguration(self):
        super(IterativeRecommender, self).readConfiguration()
        # set the reduced dimension
        self.emb_size = int(self.config['num.factors'])
        # set maximum epoch
        self.maxEpoch = int(self.config['num.max.epoch'])
        # set learning rate
        learningRate = config.OptionConf(self.config['learnRate'])
        self.lRate = float(learningRate['-init'])
        self.maxLRate = float(learningRate['-max'])
        if self.evalSettings.contains('-tf'):
            self.batch_size = int(self.config['batch_size'])
        # regularization parameter
        regular = config.OptionConf(self.config['reg.lambda'])
        self.regU,self.regI,self.regB= float(regular['-u']),float(regular['-i']),float(regular['-b'])

    def printAlgorConfig(self):
        super(IterativeRecommender, self).printAlgorConfig()
        print('Embedding Dimension:', self.emb_size)
        print('Maximum Epoch:', self.maxEpoch)
        print('Regularization parameter: regU %.3f, regI %.3f, regB %.3f' %(self.regU,self.regI,self.regB))
        print('='*80)

    def initModel(self):
        self.P = np.random.rand(len(self.data.drug), self.emb_size) / 3 # latent drug matrix
        self.Q = np.random.rand(len(self.data.molecular), self.emb_size) / 3  # latent molecular matrix
        self.loss, self.lastLoss = 0, 0

    def trainModel_tf(self):
        # initialization
        import tensorflow as tf
        self.u_idx = tf.placeholder(tf.int32, [None], name="u_idx")
        self.v_idx = tf.placeholder(tf.int32, [None], name="v_idx")
        self.r = tf.placeholder(tf.float32, [None], name="rating")
        self.U = tf.Variable(tf.truncated_normal(shape=[self.num_drugs, self.emb_size], stddev=0.0005), name='U')
        self.V = tf.Variable(tf.truncated_normal(shape=[self.num_molecular, self.emb_size], stddev=0.0005), name='V')
        self.drug_biases = tf.Variable(tf.truncated_normal(shape=[self.num_drugs, 1], stddev=0.0005), name='U')
        self.molecular_biases = tf.Variable(tf.truncated_normal(shape=[self.num_molecular, 1], stddev=0.0005), name='V') #为啥是U？
        self.drug_bias = tf.nn.embedding_lookup(self.drug_biases, self.u_idx)
        self.molecular_bias = tf.nn.embedding_lookup(self.molecular_biases, self.v_idx)
        self.drug_embedding = tf.nn.embedding_lookup(self.U, self.u_idx)
        self.molecular_embedding = tf.nn.embedding_lookup(self.V, self.v_idx)

    def updateLearningRate(self,epoch):
        if epoch > 1:
            if abs(self.lastLoss) > abs(self.loss):
                self.lRate *= 1.05
            else:
                self.lRate *= 0.5
        if self.lRate > self.maxLRate > 0:
            self.lRate = self.maxLRate

    def predictForRating(self, u, i):
        if self.data.containsdrug(u) and self.data.containsmolecular(i):
            return self.P[self.data.drug[u]].dot(self.Q[self.data.molecular[i]])
        elif self.data.containsdrug(u) and not self.data.containsmolecular(i):
            return self.data.drugMeans[u]
        elif not self.data.containsdrug(u) and self.data.containsmolecular(i):
            return self.data.molecularMeans[i]
        else:
            return self.data.globalMean

    def predictForRanking(self,u):
        'used to rank all the molecular for the drug'
        if self.data.containsdrug(u):
            return self.Q.dot(self.P[self.data.drug[u]])
        else:
            return [self.data.globalMean]*self.num_molecular   

    def isConverged(self,epoch):
        from math import isnan
        if isnan(self.loss):
            print('Loss = NaN or Infinity: current settings does not fit the recommender! Change the settings and try again!')
            exit(-1)
        deltaLoss = (self.lastLoss-self.loss)
        if self.ranking.isMainOn():
            print('%s %s epoch %d: loss = %.4f, delta_loss = %.5f learning_Rate = %.5f' \
                  % (self.modelName, self.foldInfo, epoch, self.loss, deltaLoss, self.lRate))
        else:
            measure = self.rating_performance()
            print('%s %s epoch %d: loss = %.4f, delta_loss = %.5f learning_Rate = %.5f %5s %5s' \
                  % (self.modelName, self.foldInfo, epoch, self.loss, deltaLoss, self.lRate, measure[0].strip()[:11], measure[1].strip()[:12]))
        #check if converged
        cond = abs(deltaLoss) < 1e-3
        converged = cond
        if not converged:
            self.updateLearningRate(epoch)
        self.lastLoss = self.loss
        shuffle(self.data.trainingData)
        return converged

    def rating_performance(self): #修改这里
        res = []
        for ind, entry in enumerate(self.data.testData):
            drug, molecular, rating = entry
            # predict
            prediction = self.predictForRating(drug, molecular)
            pred = self.checkRatingBoundary(prediction)
            res.append([drug,molecular,rating,pred])
        self.measure = Measure.ratingMeasure(res)
        return self.measure

    def ranking_performance(self,epoch):
        #evaluation during training
        top = self.ranking['-topN'].split(',')
        top = [int(num) for num in top]
        N = max(top)
        recList = {}
        topN_rawRes = {}
        rawRes = {}
        print('Evaluating...')
        for drug in self.data.testSet_u:
            candidates = self.predictForRanking(drug)
            # predictedfragments = denormalize(predictedfragments, self.data.rScale[-1], self.data.rScale[0])
            ratedList, ratingList = self.data.drugRated(drug) #得到drug1在训练集中相关药物的list和对应分数的list
         
            #将预测到的molecular在训练集中的score改为0
            for molecular in ratedList:
                candidates[self.data.molecular[molecular]] = 0

            ids, scores = find_k_largest(N, candidates)#20220412
            molecular_names = [self.data.id2molecular[iid] for iid in ids]
            recList[drug] = list(zip(molecular_names, scores))
            topN_rawRes[drug] = dict(zip(molecular_names, scores))

            #20220412
            all_candidates = []
            for all_id, all_score in enumerate(candidates[:]):
                all_candidates.append((all_id, all_score))
            #all_candidates.sort(key=lambda d: d[1], reverse=True) #目前算AUC的方法不用排序

            all_ids = [molecular[0] for molecular in all_candidates]
            all_molecular_names = [self.data.id2molecular[iid] for iid in all_ids]
            all_scores = [molecular[1] for molecular in all_candidates]
            rawRes[drug] = dict(zip(all_molecular_names, all_scores))
            
        measure = Measure.rankingMeasure(self.data.testSet_u, self.data.trainSet_u, recList, topN_rawRes, rawRes, [N]) 
        if len(self.bestPerformance)>0:
            count = 0
            performance = {}
            for m in measure[1:]:
                k,v = m.strip().split(':')
                performance[k]=float(v)
            for k in self.bestPerformance[1]:
                if self.bestPerformance[1][k] > performance[k]:
                    count += 1
                else:
                    count -=1
            if count<0:
                self.bestPerformance[1]=performance
                self.bestPerformance[0]=epoch+1
                self.saveModel()

        else:
            self.bestPerformance.append(epoch+1)
            performance = {}
            for m in measure[1:]:
                k,v = m.strip().split(':')
                performance[k]=float(v)
                self.bestPerformance.append(performance)
            self.saveModel()
        print('-'*120)
        print('Quick Ranking Performance '+self.foldInfo+' (Top-'+str(N)+'molecular Recommendation)')
        measure = [m.strip() for m in measure[1:]]
        print('*Current Performance*')
        print('Epoch:',str(epoch+1)+',',' | '.join(measure))
        bp = ''
        # for k in self.bestPerformance[1]:
        #     bp+=k+':'+str(self.bestPerformance[1][k])+' | '
        #bp += 'Precision'+':'+str(self.bestPerformance[1]['Precision'])+' | '
        #bp += 'Recall' + ':' + str(self.bestPerformance[1]['Recall']) + ' | '
        #bp += 'F1' + ':' + str(self.bestPerformance[1]['F1']) + ' | '
        #bp += 'MDCG' + ':' + str(self.bestPerformance[1]['NDCG']) + ' | '
        #bp += 'TopN_AUC' + ':' + str(self.bestPerformance[1]['TopN_AUC']) + ' | '
        
        bp += 'AUROC' + ':' + str(self.bestPerformance[1]['AUROC']) + ' | '
        bp += 'AUPR' + ':' + str(self.bestPerformance[1]['AUPR'])
        #bp += 'all_F1' + ':' + str(self.bestPerformance[1]['all_F1']) + ' | '
        #bp += 'acc' + ':' + str(self.bestPerformance[1]['acc'])
        print('*Best Performance* ')
        print('Epoch:',str(self.bestPerformance[0])+',',bp)
        print('-'*120)
        return measure

class DeepRecommender(IterativeRecommender):
    def __init__(self,conf,trainingSet,testSet, fold='[1]'):
        super(DeepRecommender, self).__init__(conf,trainingSet,testSet, fold)

    def readConfiguration(self):
        super(DeepRecommender, self).readConfiguration()
        self.batch_size = int(self.config['batch_size'])

    def printAlgorConfig(self):
        super(DeepRecommender, self).printAlgorConfig()

    def initModel(self):
        super(DeepRecommender, self).initModel()
        self.u_idx = tf.placeholder(tf.int32, name="u_idx")
        self.v_idx = tf.placeholder(tf.int32, name="v_idx")
        self.r = tf.placeholder(tf.float32, name="rating")
        self.drug_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_drugs, self.emb_size], stddev=0.005), name='U')#可改成输入属性特征
        self.molecular_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_molecular, self.emb_size], stddev=0.005), name='V')
        self.batch_drug_emb = tf.nn.embedding_lookup(self.drug_embeddings, self.u_idx)
        self.batch_pos_molecular_emb = tf.nn.embedding_lookup(self.molecular_embeddings, self.v_idx)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def next_batch_pairwise(self):
        shuffle(self.data.trainingData)
        batch_id = 0
        while batch_id < self.train_size:
            if batch_id + self.batch_size <= self.train_size:
                drugs = [self.data.trainingData[idx][0] for idx in range(batch_id, self.batch_size + batch_id)]
                molecular = [self.data.trainingData[idx][1] for idx in range(batch_id, self.batch_size + batch_id)]
                batch_id += self.batch_size
            else:
                drugs = [self.data.trainingData[idx][0] for idx in range(batch_id, self.train_size)]
                molecular = [self.data.trainingData[idx][1] for idx in range(batch_id, self.train_size)]
                batch_id = self.train_size

            u_idx, i_idx, j_idx = [], [], []
            molecular_list = list(self.data.molecular.keys())
            for i, drug in enumerate(drugs):
                i_idx.append(self.data.molecular[molecular[i]])
                u_idx.append(self.data.drug[drug])
                neg_molecular = choice(molecular_list)
                while neg_molecular in self.data.trainSet_u[drug]:
                    neg_molecular = choice(molecular_list)
                j_idx.append(self.data.molecular[neg_molecular])

            yield u_idx, i_idx, j_idx

    def next_batch_pointwise(self):
        batch_id=0
        while batch_id<self.train_size:
            if batch_id+self.batch_size<=self.train_size:
                drugs = [self.data.trainingData[idx][0] for idx in range(batch_id,self.batch_size+batch_id)]
                molecular = [self.data.trainingData[idx][1] for idx in range(batch_id,self.batch_size+batch_id)]
                batch_id+=self.batch_size
            else:
                drugs = [self.data.trainingData[idx][0] for idx in range(batch_id, self.train_size)]
                molecular = [self.data.trainingData[idx][1] for idx in range(batch_id, self.train_size)]
                batch_id=self.train_size
            u_idx,i_idx,y = [],[],[]
            for i,drug in enumerate(drugs):
                i_idx.append(self.data.molecular[molecular[i]])
                u_idx.append(self.data.drug[drug])
                y.append(1)
                for instance in range(4):
                    molecular_j = randint(0, self.num_molecular - 1)
                    while self.data.id2molecular[molecular_j] in self.data.trainSet_u[drug]:
                        molecular_j = randint(0, self.num_molecular - 1)
                    u_idx.append(self.data.drug[drug])
                    i_idx.append(molecular_j)
                    y.append(0)
            yield u_idx,i_idx,y

    def predictForRanking(self,u):
        'used to rank all the fragments for the drug'
        pass
        
class GraphRecommender(DeepRecommender):
    def __init__(self,conf,trainingSet,testSet,fold='[1]'):
        super(GraphRecommender, self).__init__(conf,trainingSet,testSet,fold)

    def create_joint_sparse_adjaceny(self):
        '''
        return a sparse adjacency matrix with the shape (user number + item number, user number + item number)
        '''
        n_nodes = self.num_drugs + self.num_molecular
        row_idx = [self.data.drug[pair[0]] for pair in self.data.trainingData]
        col_idx = [self.data.molecular[pair[1]] for pair in self.data.trainingData]
        drug_np = np.array(row_idx)
        molecular_np = np.array(col_idx)
        ratings = np.ones_like(drug_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (drug_np, molecular_np + self.num_drugs)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T
        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return adj_matrix

    def create_joint_sparse_adj_tensor(self):
        '''
        return a sparse tensor with the shape (user number + item number, user number + item number)
        '''
        norm_adj = self.create_joint_sparse_adjaceny()
        row,col = norm_adj.nonzero()
        indices = np.array(list(zip(row,col)))
        adj_tensor = tf.SparseTensor(indices=indices, values=norm_adj.data, dense_shape=norm_adj.shape)
        return adj_tensor

    def create_sparse_rating_matrix(self):
        '''
        return a sparse adjacency matrix with the shape (user number, item number)
        '''
        row, col, entries = [], [], []
        for pair in self.data.trainingData:
            row += [self.data.drug[pair[0]]]
            col += [self.data.molecular[pair[1]]]
            entries += [1.0/len(self.data.trainSet_u[pair[0]])]
        ratingMat = sp.coo_matrix((entries, (row, col)), shape=(self.num_drugs,self.num_molecular),dtype=np.float32)
        return ratingMat

    def create_sparse_adj_tensor(self):
        '''
        return a sparse tensor with the shape (user number, item number)
        '''
        ratingMat = self.create_sparse_rating_matrix()
        row,col = ratingMat.nonzero()
        indices = np.array(list(zip(row,col)))
        adj_tensor = tf.SparseTensor(indices=indices, values=ratingMat.data, dense_shape=ratingMat.shape)
        return adj_tensor     

class DMIRecommender(IterativeRecommender):
    def __init__(self,conf,trainingSet,testSet, fold='[1]'):
        super(DMIRecommender, self).__init__(conf,trainingSet,testSet,fold)
        self.DFload = DFload(self.config, trainingSet, testSet)

    def readConfiguration(self):
        super(DMIRecommender, self).readConfiguration()
        regular = config.OptionConf(self.config['reg.lambda'])
        self.regS = float(regular['-s']) 

    def printAlgorConfig(self):
        super(DMIRecommender, self).printAlgorConfig()
        print('DFI dataset:',abspath(self.config['DFI']))
        print('DFI relation size ','(drug count:',len(self.DFload.drug), 'fragment count:',len(self.DFload.fragment),'Relation count:'+str(len(self.DFload.DFIrelation))+')')
        print('FFI relation size ','(fragment count:',len(self.DFload.fragment), 'fragment count:',len(self.DFload.fragment),'Relation count:'+str(len(self.DFload.FFIrelation))+')')
        print('DFI Regularization parameter: regS %.3f' % (self.regS))
        print('=' * 80)


