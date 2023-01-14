from util.config import OptionConf,FileIO,DataSplit,ModelConf
from multiprocessing import Process,Manager
from time import strftime,localtime,time
import mkl


class runCom(object):
    def __init__(self,config):
        self.trainingData = []  # training data
        self.testData = []  # testData
        self.measure = []
        self.config =config
        self.TrainConfig = OptionConf(config['Task.setup'])
        if self.config.contains('evaluation.setup'):
            self.evaluation = OptionConf(config['evaluation.setup'])
            binarized = False
            bottom = 0
            if self.evaluation.contains('-b'):
                binarized = True
                bottom = float(self.evaluation['-b'])
            if self.evaluation.contains('-testSet'): #可用外部测试集进行结果评估，单独输入一个外部test文件。暂无单独test文件
                #specified testSet
                self.trainingData = FileIO.loadDataSet(config, config['Train'],binarized=binarized,threshold=bottom)
                self.testData = FileIO.loadDataSet(config, self.evaluation['-testSet'], bTest=True,binarized=binarized,threshold=bottom)

            elif self.evaluation.contains('-cv'):#交叉验证，默认5折
                #cross validation
                self.trainingData = FileIO.loadDataSet(config, config['Task'], binarized=binarized,threshold=bottom)
                #输入全部DFI文件数据作为训练数据，格式[drugId, fragmentId, float(rating)]

            elif self.evaluation.contains('-predict'):
                #cross validation
                self.trainingData = FileIO.loadDataSet(config, config['Train'],binarized=binarized,threshold=bottom)
                self.testData = FileIO.loaddrugList(self.evaluation['-predict'])

        else:
            print('Wrong configuration of evaluation!')
            exit(-1)

    def execute(self):
        from model import model
        # 20220408
        if self.evaluation.contains('-cv'):
            k = int(self.evaluation['-cv'])
            if k < 2 or k > 10: #limit to 2-10 fold cross validation
                print("k for cross-validation should not be greater than 10 or less than 2")
                exit(-1)
            mkl.set_num_threads(max(1,mkl.get_max_threads()//k))
            #create the manager for communication among multiple processes
            manager = Manager()
            mDict = manager.dict()
            i = 1
            tasks = []
            binarized = False
            if self.evaluation.contains('-b'):
                binarized = True
            for train,test in DataSplit.crossValidation(self.trainingData,k,binarized=binarized): #对数据进行切分，进行交叉验证
                fold = '['+str(i)+']'
                recommender = self.config['model.name'] + "(self.config,train,test,fold)"
               #create the process
                p = Process(target=run,args=(mDict,eval(recommender),i))#eval(recommender)为返回recommender的结果。
                tasks.append(p)
                i+=1
            #start the processes
            for p in tasks:
                p.start()
                if not self.evaluation.contains('-p'):
                    p.join()
            #wait until all processes are completed
            if self.evaluation.contains('-p'):
                for p in tasks:
                    p.join()
            #compute the average error of k-fold cross validation
            self.measure = [dict(mDict)[i] for i in range(1,k+1)]
            res = []
            for i in range(len(self.measure[0])):
                if self.measure[0][i][:3] == 'Top':
                    res.append(self.measure[0][i])
                    continue
                measure = self.measure[0][i].split(':')[0]
                #print('measure:', measure,"\n")
                total = 0
                for j in range(k):
                    total += float(self.measure[j][i].split(':')[1])
                res.append(measure + ':' + str(total / k) + '\n')
                #print('res:', res,"\n")
            #print('res:', len(res),"\n")
            #output result
            currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
            outDir = OptionConf(self.config['output.setup'])['-dir']
            fileName = self.config['model.name'] +'@'+currentTime+'-'+str(k)+'-fold-cv' + '.txt'
            FileIO.writeFile(outDir,fileName,res)
            print('The result of %d-fold cross validation:\n%s' %(k,''.join(res)))

        else:
            recommender = self.config['model.name'] + '(self.config,self.trainingData,self.testData)'
            eval(recommender).execute()


def run(measure,algor,order):
    measure[order] = algor.execute()
    
    
if __name__ == '__main__':
    #import time
    #s = time.time()
    s = time()
    try:
        conf = ModelConf('model.conf')
    except KeyError:
        print('wrong num!')
        exit(-1)
    recSys = runCom(conf)
    recSys.execute()
    #e = time.time()
    e = time()
    print("Running time: %f s" % (e - s))