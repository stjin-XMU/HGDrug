import os.path
from numpy.linalg import norm
from numba import jit
from random import random, sample
from re import compile,findall,split
import logging
import os
import tensorflow as tf
import math
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, accuracy_score

class ModelConf(object):
    def __init__(self,fileName):
        self.config = {}
        self.readConfiguration(fileName)

    def __getitem__(self, item):
        if not self.contains(item):
            print('parameter '+item+' is invalid!')
            exit(-1)
        return self.config[item]

    def contains(self,key):
        return key in self.config

    def readConfiguration(self,file):
        if not os.path.exists(file):
            print('config file is not found!')
            raise IOError
        with open(file) as f:
            for ind,line in enumerate(f):
                if line.strip()!='':
                    try:
                        key,value=line.strip().split('=')
                        self.config[key]=value
                    except ValueError:
                        print('config file is not in the correct format! Error Line:%d'%(ind))

class OptionConf(object):
    def __init__(self,content):
        self.line = content.strip().split(' ')
        self.options = {}
        self.mainOption = False
        if self.line[0] == 'on':
            self.mainOption = True
        elif self.line[0] == 'off':
            self.mainOption = False
        for i,item in enumerate(self.line):
            if (item.startswith('-') or item.startswith('--')) and  not item[1:].isdigit():
                ind = i+1
                for j,sub in enumerate(self.line[ind:]):
                    if (sub.startswith('-') or sub.startswith('--')) and  not sub[1:].isdigit():
                        ind = j
                        break
                    if j == len(self.line[ind:])-1:
                        ind=j+1
                        break
                try:
                    self.options[item] = ' '.join(self.line[i+1:i+1+ind])
                except IndexError:
                    self.options[item] = 1

    def __getitem__(self, item):
        if not self.contains(item):
            print('parameter '+item+' is invalid!')
            exit(-1)
        return self.options[item]

    def keys(self):
        return self.options.keys()

    def isMainOn(self):
        return self.mainOption

    def contains(self,key):
        return key in self.options


class DataSplit(object):

    def __init__(self):
        pass

    @staticmethod
    def dataSplit(data,test_ratio = 0.3,output=False,path='./',order=1,binarized = False):
        if test_ratio>=1 or test_ratio <=0:
            test_ratio = 0.3
        testSet = []
        trainingSet = []
        for entry in data:
            if random() < test_ratio:
                if binarized:
                    if entry[2]:
                        testSet.append(entry)
                else:
                    testSet.append(entry)
            else:
                trainingSet.append(entry)
        if output:
            FileIO.writeFile(path,'testSet['+str(order)+']',testSet)
            FileIO.writeFile(path, 'trainingSet[' + str(order) + ']', trainingSet)
        return trainingSet,testSet

    @staticmethod
    def crossValidation(data,k,output=False,path='./',order=1,binarized=False):
        if k<=1 or k>10:
            k=3
        for i in range(k):
            trainingSet = []
            testSet = []
            for ind,line in enumerate(data):
                if ind%k == i:
                    if binarized:
                        if line[2]:
                            testSet.append(line[:])
                    else:
                        testSet.append(line[:])
                else:
                    trainingSet.append(line[:])
            yield trainingSet,testSet



class FileIO(object): #定义输入输出处理的类
    def __init__(self):
        pass

    # @staticmethod
    # def writeFile(filePath,content,op = 'w'):
    #     reg = compile('(.+[/|\\\]).+')
    #     dirs = findall(reg,filePath)
    #     if not os.path.exists(filePath):
    #         os.makedirs(dirs[0])
    #     with open(filePath,op) as f:
    #         f.write(str(content))

    @staticmethod
    def writeFile(dir,file,content,op = 'w'):
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(dir+file,op) as f:
            f.writelines(content)

    @staticmethod
    def deleteFile(filePath):
        if os.path.exists(filePath):
            remove(filePath)

    @staticmethod
    def loadDataSet(conf, file, bTest=False,binarized = False, threshold = 3.0): #train data load
        trainingData = []
        testData = []
        TrainConfig = OptionConf(conf['Task.setup'])
        if not bTest:
            print('loading training data...')
        else:
            print('loading test data...')
        with open(file) as f:
            traindatas = f.readlines()
        # ignore the headline
        if TrainConfig.contains('-header'):
            traindatas = traindatas[1:]
        # order of the columns
        order = TrainConfig['-columns'].strip().split()
        delim = ' |,|\t'
        if TrainConfig.contains('-delim'):
            delim=TrainConfig['-delim']
        for lineNo, line in enumerate(traindatas):
            lines = split(delim,line.strip())
            if not bTest and len(order) < 2:
                print('The rating file is not in a correct format. Error: Line num %d' % lineNo)
                exit(-1)
            try:
                drugId = lines[int(order[0])]
                molecularId = lines[int(order[1])]
                if len(order)<3:
                    rating = 1 #default value
                else:
                    rating  = lines[int(order[2])]
                if binarized:
                    rating = 1
            except ValueError:
                print('Error! Have you added the option -header to the rating.setup?')
                exit(-1)
            if bTest:
                testData.append([drugId, molecularId, float(rating)])
            else:
                trainingData.append([drugId, molecularId, float(rating)])
        if bTest:
            return testData
        else:
            return trainingData

    @staticmethod
    def loaddrugList(filepath):
        drugList = []
        print('loading drug List...')
        with open(filepath) as f:
            for line in f:
                drugList.append(line.strip().split()[0])
        return drugList

    @staticmethod
    def loadDDship(conf, filePath): #d-d
        DDIConfig = OptionConf(conf['DDI.setup'])
        relation = []
        with open(filePath) as f:
            relations = f.readlines()
            # ignore the headline
        if DDIConfig.contains('-header'):
            relations = relations[1:]
        # order of the columns
        order = DDIConfig['-columns'].strip().split()
        for lineNo, line in enumerate(relations):
            lines = split(' |,|\t', line.strip())
            if len(order) < 2:
                print('The DDI file is not in a correct format. Error: Line num %d' % lineNo)
                exit(-1)
            drugId1 = lines[int(order[0])]
            drugId2 = lines[int(order[1])]
            weight = 1
            relation.append([drugId1, drugId2, weight])
        return relation

    @staticmethod
    def loadDFship(conf, filePath): #d-f
        DFIConfig = OptionConf(conf['DFI.setup'])
        relation = []
        with open(filePath) as f:
            relations = f.readlines()
            # ignore the headline
        if DFIConfig.contains('-header'):
            relations = relations[1:]
        # order of the columns
        order = DFIConfig['-columns'].strip().split()
        for lineNo, line in enumerate(relations):
            lines = split(' |,|\t', line.strip())
            if len(order) < 2:
                print('The DDI file is not in a correct format. Error: Line num %d' % lineNo)
                exit(-1)
            drugId = lines[int(order[0])]
            fragmentID = lines[int(order[1])]
            weight = 1
            relation.append([drugId, fragmentID, weight])
        return relation

    @staticmethod
    def loadFFship(conf, filePath): #f-f
        FFIConfig = OptionConf(conf['FFI.setup'])
        relation = []
        with open(filePath) as f:
            relations = f.readlines()
            # ignore the headline
        if FFIConfig.contains('-header'):
            relations = relations[1:]
        # order of the columns
        order =FFIConfig['-columns'].strip().split()
        for lineNo, line in enumerate(relations):
            lines = split(' |,|\t', line.strip())
            if len(order) < 2:
                print('The FFI file is not in a correct format. Error: Line num %d' % lineNo)
                exit(-1)
            fragmentID1 = lines[int(order[0])]
            fragmentID2= lines[int(order[1])]
            weight = 1
            relation.append([fragmentID1, fragmentID2, weight])
        return relation
    
    @staticmethod
    def loadDTship(conf, filePath): #d-t
        DTIConfig = OptionConf(conf['DTI.setup'])
        relation = []
        with open(filePath) as f:
            relations = f.readlines()
            # ignore the headline
        if DTIConfig.contains('-header'):
            relations = relations[1:]
        # order of the columns
        order = DTIConfig['-columns'].strip().split()
        for lineNo, line in enumerate(relations):
            lines = split(' |,|\t', line.strip())
            if len(order) < 2:
                print('The DDI file is not in a correct format. Error: Line num %d' % lineNo)
                exit(-1)
            drugId = lines[int(order[0])]
            targetID = lines[int(order[1])]
            weight = 1
            relation.append([drugId, targetID, weight])
        return relation
        
    @staticmethod
    def loadDDiship(conf, filePath): #d-di
        DDiIConfig = OptionConf(conf['DDiI.setup'])
        relation = []
        with open(filePath) as f:
            relations = f.readlines()
            # ignore the headline
        if DDiIConfig.contains('-header'):
            relations = relations[1:]
        # order of the columns
        order = DDiIConfig['-columns'].strip().split()
        for lineNo, line in enumerate(relations):
            lines = split(' |,|\t', line.strip())
            if len(order) < 2:
                print('The DDI file is not in a correct format. Error: Line num %d' % lineNo)
                exit(-1)
            drugId = lines[int(order[0])]
            diseaseID = lines[int(order[1])]
            weight = 1
            relation.append([drugId, diseaseID, weight])
        return relation
    
    @staticmethod
    def loadDSship(conf, filePath): #d-s
        DSIConfig = OptionConf(conf['DSI.setup'])
        relation = []
        with open(filePath) as f:
            relations = f.readlines()
            # ignore the headline
        if DSIConfig.contains('-header'):
            relations = relations[1:]
        # order of the columns
        order = DSIConfig['-columns'].strip().split()
        for lineNo, line in enumerate(relations):
            lines = split(' |,|\t', line.strip())
            if len(order) < 2:
                print('The DDI file is not in a correct format. Error: Line num %d' % lineNo)
                exit(-1)
            drugId = lines[int(order[0])]
            sideeffectID = lines[int(order[1])]
            weight = 1
            relation.append([drugId, sideeffectID, weight])
        return relation
    
class Log(object):
    def __init__(self,module,filename):
        self.logger = logging.getLogger(module)
        self.logger.setLevel(level=logging.INFO)
        if not os.path.exists('./log/'):
            os.makedirs('./log/')
        handler = logging.FileHandler('./log/'+filename+'.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def add(self,text):
        self.logger.info(text)

def bpr_loss(drug_emb,pos_molecular_emb,neg_molecular_emb):
    score = tf.reduce_sum(tf.multiply(drug_emb, pos_molecular_emb), 1) - tf.reduce_sum(tf.multiply(drug_emb, neg_molecular_emb), 1)
    loss = -tf.reduce_sum(tf.math.log(tf.sigmoid(score)+10e-8))
    return loss

class Measure(object):
    def __init__(self):
        pass


    @staticmethod
    def rankingMeasure(origin, trainSet, res, topN_rawRes, rawRes, N):#20220411
        measure = []
        n=N
        measure.append('Top ' + str(n) + '\n')
        AUROC, AUPR = Measure.all_AUC_bla(origin, trainSet, rawRes)
        measure.append('AUROC:' + str(AUROC) + '\n')
        measure.append('AUPR:' + str(AUPR) + '\n')

        return measure

    @staticmethod
    def all_AUC_bla(origin, trainSet, rawRes): #随机选取1：1的负样本的比例算AUC和AUPR
        label_pos = []
        pre_pos = []
        label_neg = []
        pre_neg = []
        for drug in rawRes:
            for molecular in rawRes[drug]: 
                #if fragment not in trainSet[drug]:#避免全局AUC计算时，默认训练集中出现过的正样本被当作负样本。
                if molecular in origin[drug]:
                    label_pos.append(1)
                    pre_pos.append(rawRes[drug][molecular])
                elif molecular not in trainSet[drug]:#避免全局AUC计算时，默认训练集中出现过的正样本被当作负样本。
                    label_neg.append(0)
                    pre_neg.append(rawRes[drug][molecular])
        choose_pre_neg = sample(pre_neg, len(label_pos)) #随机选取1：1的负样本的比例算AUC和AUPR，test中交互有18567
        pre = pre_pos + choose_pre_neg
        label = label_pos + (len(pre)-len(label_pos))*[0] 
        fpr, tpr, th = roc_curve(label, pre , pos_label=1)
        precision, recall, threshold = precision_recall_curve(label, pre)
        AUROC = auc(fpr, tpr)
        AUPR =  auc(recall,precision)

        precision_f1 = sum(precision) / len(precision)
        recall_f1 = sum(recall) / len(recall)
        
        y_preds = []
        acc_threshold = sum(pre) / len (pre)
        for i in pre:
            if i >= acc_threshold:
                y_preds.append(1)
            else:
                y_preds.append(0)

        F1 = 2 * precision_f1 * recall_f1 / (precision_f1 + recall_f1)
        acc = accuracy_score(label, y_preds)
        print('rawRes', len(rawRes))
        print('choose_pre_neg', len(choose_pre_neg))
        print('label', len(label))
        print('pre', len(pre))
        print('threshold', len(threshold))
        return AUROC, AUPR

def l1(x):
    return norm(x,ord=1)

def l2(x):
    return norm(x)

def common(x1,x2):
    # find common DFI
    common = (x1!=0)&(x2!=0)
    new_x1 = x1[common]
    new_x2 = x2[common]
    return new_x1,new_x2

def cosine_sp(x1,x2):
    'x1,x2 are dicts,this version is for sparse representation'
    total = 0
    denom1 = 0
    denom2 =0
    try:
        for k in x1:
            if k in x2:
                total+=x1[k]*x2[k]
                denom1+=x1[k]**2
                denom2+=x2[k]**2
        return total/(sqrt(denom1) * sqrt(denom2))
    except ZeroDivisionError:
        return 0

def euclidean_sp(x1,x2):
    'x1,x2 are dicts,this version is for sparse representation'
    total = 0
    try:
        for k in x1:
            if k in x2:
                total+=x1[k]**2-x2[k]**2
        return 1/total
    except ZeroDivisionError:
        return 0

def cosine(x1,x2):
    #find common DFI
    #new_x1, new_x2 = common(x1,x2)
    #compute the cosine similarity between two vectors
    sum = x1.dot(x2)
    denom = sqrt(x1.dot(x1)*x2.dot(x2))
    try:
        return sum/denom
    except ZeroDivisionError:
        return 0

    #return cosine_similarity(x1,x2)[0][0]

def pearson_sp(x1,x2):
    total = 0
    denom1 = 0
    denom2 = 0
    overlapped=False
    try:
        mean1 = sum(x1.values())/len(x1)
        mean2 = sum(x2.values()) /len(x2)
        for k in x1:
            if k in x2:
                total += (x1[k]-mean1) * (x2[k]-mean2)
                denom1 += (x1[k]-mean1) ** 2
                denom2 += (x2[k]-mean2) ** 2
                overlapped=True
        return total/ (sqrt(denom1) * sqrt(denom2))
    except ZeroDivisionError:
        if overlapped:
            return 1
        return 0

def euclidean(x1,x2):
    #find common DFI
    new_x1, new_x2 = common(x1, x2)
    #compute the euclidean between two vectors
    diff = new_x1-new_x2
    denom = sqrt((diff.dot(diff)))
    try:
        return 1/denom
    except ZeroDivisionError:
        return 0


def pearson(x1,x2):
    #find common DFI
    #new_x1, new_x2 = common(x1, x2)
    #compute the pearson similarity between two vectors
    #ind1 = new_x1 > 0
    #ind2 = new_x2 > 0
    try:
        mean_x1 = x1.sum()/len(x1)
        mean_x2 = x2.sum()/len(x2)
        new_x1 = x1 - mean_x1
        new_x2 = x2 - mean_x2
        sum = new_x1.dot(new_x2)
        denom = sqrt((new_x1.dot(new_x1))*(new_x2.dot(new_x2)))
        return sum/denom
    except ZeroDivisionError:
        return 0


def similarity(x1,x2,sim):
    if sim == 'pcc':
        return pearson_sp(x1,x2)
    if sim == 'euclidean':
        return euclidean_sp(x1,x2)
    else:
        return cosine_sp(x1, x2)


def normalize(vec,maxVal,minVal):
    'get the normalized value using min-max normalization'
    if maxVal > minVal:
        return (vec-minVal)/(maxVal-minVal)
    elif maxVal==minVal:
        return vec/maxVal
    else:
        print('error... maximum value is less than minimum value.')
        raise ArithmeticError

def sigmoid(val):
    return 1/(1+exp(-val))


def denormalize(vec,maxVal,minVal):
    return minVal+(vec-0.01)*(maxVal-minVal)

@jit(nopython=True)
def find_k_largest(K,candidates):
    n_candidates = []
    for iid,score in enumerate(candidates[:K]):
        n_candidates.append((iid, score))
    n_candidates.sort(key=lambda d: d[1], reverse=True)
    k_largest_scores = [molecular[1] for molecular in n_candidates]
    ids = [molecular[0] for molecular in n_candidates]
    # find the N biggest scores
    for iid,score in enumerate(candidates):
        ind = K
        l = 0
        r = K - 1
        if k_largest_scores[r] < score:
            while r >= l:
                mid = int((r - l) / 2) + l
                if k_largest_scores[mid] >= score:
                    l = mid + 1
                elif k_largest_scores[mid] < score:
                    r = mid - 1
                if r < l:
                    ind = r
                    break
        # move the fragments backwards
        if ind < K - 2:
            k_largest_scores[ind + 2:] = k_largest_scores[ind + 1:-1]
            ids[ind + 2:] = ids[ind + 1:-1]
        if ind < K - 1:
            k_largest_scores[ind + 1] = score
            ids[ind + 1] = iid
    return ids,k_largest_scores

class new_sparseMatrix():
    'matrix used to store raw data'
    def __init__(self,triple):
        self.matrix_drug = {}
        self.matrix_molecular = {}
        for molecular in triple:
            if molecular[0] not in self.matrix_drug:
                self.matrix_drug[molecular[0]] = {}
            if molecular[1] not in self.matrix_molecular:
                self.matrix_molecular[molecular[1]] = {}
            self.matrix_drug[molecular[0]][molecular[1]]=molecular[2]
            self.matrix_molecular[molecular[1]][molecular[0]]=molecular[2]
        self.elemNum = len(triple)
        self.size = (len(self.matrix_drug),len(self.matrix_molecular))

    def sRow(self,r):
        if r not in self.matrix_drug:
            return {}
        else:
            return self.matrix_drug[r]

    def sCol(self,c):
        if c not in self.matrix_molecular:
            return {}
        else:
            return self.matrix_molecular[c]

    def row(self,r):
        if r not in self.matrix_drug:
            return np.zeros((1,self.size[1]))
        else:
            array = np.zeros((1,self.size[1]))
            ind = list(self.matrix_drug[r].keys())
            val = list(self.matrix_drug[r].values())
            array[0][ind] = val
            return array

    def col(self,c):
        if c not in self.matrix_molecular:
            return np.zeros((1,self.size[0]))
        else:
            array = np.zeros((1,self.size[0]))
            ind = list(self.matrix_molecular[c].keys())
            val = list(self.matrix_molecular[c].values())
            array[0][ind] = val
            return array
    def elem(self,r,c):
        if not self.contains(r,c):
            return 0
        return self.matrix_drug[r][c]

    def contains(self,r,c):
        if r in self.matrix_drug and c in self.matrix_drug[r]:
            return True
        return False

    def elemCount(self):
        return self.elemNum

    def size(self):
        return self.size
