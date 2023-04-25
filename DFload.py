import numpy as np
from util.config import ModelConf,OptionConf, FileIO
import random
from collections import defaultdict
from util.config import new_sparseMatrix

class DFload(object):
    'data access control'
    def __init__(self, config, trainingSet, testSet):
        self.config = config
        self.evalSettings = OptionConf(self.config['evaluation.setup'])
        self.drugMeans = {} #mean values of drugs's DFI
        #self.fragmentMeans = {} #mean values of fragments's DFI
        self.drug = {}
        self.id2drug = {}
        self.fragment = {}
        self.id2fragment = {}
        self.molecular = {}
        self.id2molecular = {}
        self.target = {}
        self.id2target = {}
        self.disease = {}
        self.id2disease = {}
        self.sideeffect = {}
        self.id2sideeffect = {}
        self.globalMean = 0
        self.trainSet_u = defaultdict(dict)
        self.trainSet_i = defaultdict(dict)
        self.testSet_u = defaultdict(dict) #test set in the form of [drug][drug]=rating
        self.testSet_i = defaultdict(dict) #test set in the form of [fragment][drug]=rating
        self.rScale = [] #rating scale
        self.trainingData = trainingSet[:]
        self.testData = testSet[:]
        self.DFIrelation = []
        self.FFIrelation = []
        self.DDIrelation = []
        self.DTIrelation = []
        self.DDiIrelation = []
        self.DSIrelation = []


        if config.contains('DFI'): #d-f
            self.DFIConfig = OptionConf(self.config['DFI.setup'])
            self.DFIrelation = FileIO.loadDFship(config,self.config['DFI'])

        if config.contains('FFI'): #f-f
            self.FFIConfig = OptionConf(self.config['FFI.setup'])
            self.FFIrelation = FileIO.loadFFship(config,self.config['FFI'])

        if config.contains('DDI'): #d-d
            self.DDIConfig = OptionConf(self.config['DDI.setup'])
            self.DDIrelation = FileIO.loadDDship(config,self.config['DDI'])
            
        if config.contains('DTI'): #d-t
            self.DTIConfig = OptionConf(self.config['DTI.setup'])
            self.DTIrelation = FileIO.loadDTship(config,self.config['DTI'])

        if config.contains('DDiI'): #d-di
            self.DDiIConfig = OptionConf(self.config['DDiI.setup'])
            self.DDiIrelation = FileIO.loadDDiship(config,self.config['DDiI'])

        if config.contains('DSI'): #d-s
            self.DSIConfig = OptionConf(self.config['DSI.setup'])
            self.DSIrelation = FileIO.loadDSship(config,self.config['DSI'])
            
        self.dfi = defaultdict(dict)
        self.fdi = defaultdict(dict)
        self.__generateDict()  #map drug and fragment names to id
        #self.DFIMatrix = self.__generateDFISet()
        #self.FFIMatrix = self.__generateFFISet()
        #self.relationDFI = []
        self.__generateSet()
        #self.__computefragmentMean()
        self.__computedrugMean()
        self.__globalAverage()
        
        if self.evalSettings.contains('-cold'):
            #evaluation on cold-start drugs
            self.__cold_start_test()


    def __generateDict(self):
        if self.config['Task.name'] == 'DrugDrug':
            for i,entry in enumerate(self.trainingData):
                drugName1,drugName2,rating = entry
                # order the drug
                if drugName1 not in self.drug:
                    self.drug[drugName1] = len(self.drug)
                    self.id2drug[self.drug[drugName1]] = drugName1
                # order the drug
                if drugName2 not in self.drug:
                    self.drug[drugName2] = len(self.drug)
                    self.id2drug[self.drug[drugName2]] = drugName2
                self.molecular = self.drug
                self.id2molecular = self.id2drug
        else:
            for i,entry in enumerate(self.trainingData):
                drugName,molecularName,rating = entry
                # order the drug
                if drugName not in self.drug:
                    self.drug[drugName] = len(self.drug)
                    self.id2drug[self.drug[drugName]] = drugName
                # order the disease
                if molecularName not in self.molecular:
                    self.molecular[molecularName] = len(self.molecular)
                    self.id2molecular[self.molecular[molecularName]] = molecularName   
  
        for line in self.DFIrelation: #d-f
            drugName,fragmentName,weight = line
            if fragmentName not in self.fragment:
                self.fragment[fragmentName] = len(self.fragment)
                self.id2fragment[self.fragment[fragmentName]] = fragmentName

        for line in self.FFIrelation: #f-f
            fragmentName1,fragmentName2,weight = line
            if fragmentName1 not in self.fragment:
                self.fragment[fragmentName1] = len(self.fragment)
                self.id2fragment[self.fragment[fragmentName1]] = fragmentName1
            if fragmentName2 not in self.fragment:
                self.fragment[fragmentName2] = len(self.fragment)
                self.id2fragment[self.fragment[fragmentName2]] = fragmentName2
                
        for line in self.DTIrelation: #d-t
            drugName,targetName,weight = line
            if targetName not in self.target:
                self.target[targetName] = len(self.target)
                self.id2target[self.target[targetName]] = targetName

        for line in self.DDiIrelation: #d-di
            drugName,diseaseName,weight = line
            if diseaseName not in self.disease:
                self.disease[diseaseName] = len(self.disease)
                self.id2disease[self.disease[diseaseName]] = diseaseName

        for line in self.DSIrelation: #d-s
            drugName,sideeffectName,weight = line
            if sideeffectName not in self.sideeffect:
                self.sideeffect[sideeffectName] = len(self.sideeffect)
                self.id2sideeffect[self.sideeffect[sideeffectName]] = sideeffectName

    def __generateSet(self):
        scale = set()
        #if validation is conducted, we sample the training data at a given probability to form the validation set,
        #and then replacing the test data with the validation data to tune parameters.
        if self.evalSettings.contains('-val'):
            random.shuffle(self.trainingData)
            separation = int(self.elemCount()*float(self.evalSettings['-val']))
            self.testData = self.trainingData[:separation]
            self.trainingData = self.trainingData[separation:]
        for i,entry in enumerate(self.trainingData):
            drugName,molecularName,rating = entry
            # makes the rating within the range [0, 1].
            #rating = normalize(float(rating), self.rScale[-1], self.rScale[0])
            #self.trainingData[i][2] = rating
            self.trainSet_u[drugName][molecularName] = rating
            self.trainSet_i[molecularName][drugName] = rating
            scale.add(float(rating))

        self.rScale = list(scale)
        self.rScale.sort()
        for entry in self.testData:
            if self.evalSettings.contains('-predict'):
                self.testSet_u[entry]={}
            else:
                drugName,molecularName,rating = entry
                self.testSet_u[drugName][molecularName] = rating
                self.testSet_i[molecularName][drugName]  = rating

    def __generateDFISet(self):
        triple = []
        for line in self.DFIrelation:
            drugId,fragmentId,weight = line
            if drugId in self.drug:
            #add relations to dict
                self.dfi[drugId][fragmentId] = weight
                self.fdi[fragmentId][drugId] = weight
                triple.append([self.drug[drugId], self.fragment[fragmentId], weight])
        return new_sparseMatrix(triple)

    def __generateFFISet(self):
        triple = []
        for line in self.FFIrelation:
            fragmentId1,fragmentId2,weight = line
            #add relations to dict
            self.dfi[fragmentId1][fragmentId2] = weight
            self.fdi[fragmentId2][fragmentId1] = weight
            triple.append([self.fragment[fragmentId1], self.fragment[fragmentId2], weight])
        return new_sparseMatrix(triple)

    def __globalAverage(self):
        total = sum(self.drugMeans.values())
        if total==0:
            self.globalMean = 0
        else:
            self.globalMean = total/len(self.drugMeans)

    def __computedrugMean(self):
        for u in self.drug:
            if len(self.trainSet_u[u]) == 0:
                self.drugMeans[u] = 0
            else:
                self.drugMeans[u] = sum(self.trainSet_u[u].values())/len(self.trainSet_u[u]) #len(self.trainSet_u[u]会为0，为什么

    def __computefragmentMean(self):
        for c in self.fragment:
            self.fragmentMeans[c] = sum(self.trainSet_i[c].values())/len(self.trainSet_i[c])

    def getdrugId(self,u):
        if u in self.drug:
            return self.drug[u]

    def getfragmentId(self,i):
        if i in self.fragment:
            return self.fragment[i]

    def trainingSize(self):
        train_drug = {}
        train_molecular = {}
        for i,entry in enumerate(self.trainingData):
            drugName,molecularName,rating = entry
            # order the drug1
            if drugName not in train_drug:
                train_drug[drugName] = len(train_drug)
            # order the disease
            if molecularName not in train_molecular:
                train_molecular[molecularName] = len(train_molecular)
        return (len(train_drug),len(train_molecular),len(self.trainingData))
    
    def testSize(self):
        return (len(self.testSet_u),len(self.testSet_i),len(self.testData))

    def FragmentSize(self):
        return (len(self.fragment))

    def DrugSize(self):
        return (len(self.drug))

    def molecularSize(self):
        return (len(self.molecular))
    
    def DiseaseSize(self):
        return (len(self.disease))

    def TargetSize(self):
        return (len(self.target))    

    def sideeffectSize(self):
        return (len(self.sideeffect))
 
    def contains(self,u,i):
        'whether drug1 u rated drug2 i'
        if u in self.drug and i in self.trainSet_u[u]:
            return True
        else:
            return False

    def containsdrug(self,u):
        'whether drug is in training set'
        if u in self.drug:
            return True
        else:
            return False
        
    def containsmolecular(self,u):
        'whether molecular is in training set'
        if u in self.molecular:
            return True
        else:
            return False

    def containsfragment(self,i):
        'whether fragment is in training set'
        if i in self.fragment:
            return True
        else:
            return False

    def drugRated(self,u): #return的是药物u的相关药物的list和分数list
        return list(self.trainSet_u[u].keys()),list(self.trainSet_u[u].values())

    def row(self,u):
        k,v = self.drugRated(u)
        vec = np.zeros(len(self.drug))
        #print vec
        for pair in zip(k,v):
            iid = self.drug[pair[0]]
            vec[iid]=pair[1]
        return vec

    def matrix(self): #存储了drug-drug得到的rating分数的矩阵
        m = np.zeros((len(self.drug),len(self.drug)))
        for u in self.drug:
            k, v = self.drugRated(u)
            vec = np.zeros(len(self.drug))
            # print vec
            for pair in zip(k, v):
                iid = self.drug[pair[0]]
                vec[iid] = pair[1]
            m[self.drug[u]]=vec
        return m
    # def row(self,u):
    #     return self.trainingMatrix.row(self.getdrugId(u))
    #
    # def col(self,c):
    #     return self.trainingMatrix.col(self.getfragmentId(c))

    def sRow(self,u):
        return self.trainSet_u[u]

    def sCol(self,c):
        return self.trainSet_i[c]

    def rating(self,u,c):
        if self.contains(u,c):
            return self.trainSet_u[u][c]
        return -1

    def DFIcale(self):
        return (self.rScale[0],self.rScale[1])

    def elemCount(self):
        return len(self.trainingData)
