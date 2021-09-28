import numpy as np
import pandas as pd
import re

#cart代码
class Cart():
    def __init__(self):
        self.label_x = []       #列表用来储存预测的标签
    def gini(self,x):
        '''
        :param x: x为输入数据
        :return: 数据x 的基尼系数
        '''
        n = len(x)              #n表示数据集的长度
        gini_D = 1 - np.sum(
            [np.square(
                np.sum(
                    x.iloc[:,-1]==np.unique(x[x.columns[x.shape[1]-1]])[i])/n)
                for i in range(len(np.unique(x.iloc[:,-1])))])
        return gini_D
    def gini_a(self, e, x,a):
        '''
        :param e: e 为阈值,大于e 的为一类，小于e 的为另一类
        :param x: x 为输入数据
        :param a: a 为输入数据x的列名
        :return: 数值特征a下数据集x的基尼系数
        '''
        n1 = len(x)
        D1 = np.sum(x[a]<=e)
        D2 = np.sum(x[a]>e)
        gini_a = ((D1/n1)*self.gini(x[x[a]<=e])) + ((D2/n1)*self.gini(x[x[a]>e]))
        return gini_a
    def gini_b(self,e,x,a):         #计算字符特征a下数据集的基尼系数,e为阈值，x为dataframe类型的数据,a为列名
        n2 = len(x)
        D1 = np.sum(x[a]==e)
        D2 = np.sum(x[a]!=e)
        gini_b = ((D1/n2)*self.gini(x[x[a]==e])) + ((D2/n2)*self.gini(x[x[a]!=e]))

    def SplitDataSet(self,x, a, e):
        '''
        输入：数据集，数据集中某一特征列，该特征列中的某个取值
        功能：将数据集按特征列的某一取值换分为左右两个子数据集
        输出：左右子数据集
        '''
        matLeft = x[x[a] <= e]
        matRight = x[x[a] > e]
        return matLeft, matRight

    #寻找最小基尼系数所对应的特征和阈值
    def search(self,x):           #x为划分之后的数据集，含有特征和标签
        list=[0,0,1]                #给列表赋初始值
        for i in range(x.shape[1]-1):             #对于每一个特征
            if (type(x.iloc[0,i])== np.float64):              #特征为连续值时
                c = x[x.columns[i]]                        #c 为一列数据
                c = c.sort_values().values             #排序
                for j in range(len(c)-1):       #不同阈值
                    e = (c[j]+c[j+1])/2           #阈值
                    try:
                        gini_i = self.gini_a(e,x,x.columns[i])      #某一特征下以e为阈值时的基尼系数
                    except:
                        print('gini_a error')
                    if list[2]<=gini_i:         #判断基尼系数是不是最小的,如果是则继续，否则保存新的值
                        continue
                    else:
                        # 制作一个列表，第一个元素是特征，第二个是阈值，第三个是基尼系数
                        list = []
                        list.append(x.columns[i])
                        list.append(e)
                        list.append(gini_i)
            elif (type(x.iloc[0,i])== np.str_):                             #特征为离散数据时
                for j in np.unique(x.iloc[:,i].values):
                    gini_j = self.gini_b(j,x,x.columns[i])
                    if list[2]<=gini_i:         #判断基尼系数是不是最小的,如果是则继续，否则保存新的值
                        continue
                    else:
                        # 制作一个列表，第一个元素是特征，第二个是阈值，第三个是基尼系数
                        list = []
                        list.append(x.columns[i])
                        list.append(j)
                        list.append(gini_i)


        return list[0],list[1]


    def create_tree(self,x,least_sample_number=1,least_gini=0.1):
        '''

        :param x: 输入的数据
        :param least_sample_number: children node最少的样本数，小于时生成leaf node
        :param least_gini: children node 最小的基尼系数，小于时生成leaf node
        :return: 生成好的决策树
        '''
        if (len(x)<=least_sample_number) or (len(np.unique(x.iloc[:,-1]))==1):
            return x.iloc[0,-1]               #输出label为叶子节点
        a , e = self.search(x)
        matleft,matright = self.SplitDataSet(x,a,e)
        cart_tree={}
        cart_tree[a] = {}
        cart_tree[a]['<=' + str(e) + 'contains' + str(len(matleft))] = self.create_tree(matleft)
        cart_tree[a]['>' + str(e) + 'contains' + str(len(matright))] = self.create_tree(matright)
        return cart_tree
    def predict(self,tree,x):       #tree为树模型，x为待预测数据的样本
        self.label_x = []

        #编写正则，替换出树中的阈值数字
        regex = re.compile('<=|>(.+)contains')
        for j in range(len(x)):         #分别预测每个样本的标签
            def pre_one_label(tree):
                for i in tree:
                    value = x.loc[j,i]
                    l=[]
                    for a in tree[i]:
                        l.append(a)

                    epison = float(re.findall(regex,a)[0])
                    if value<=epison:       #小于阈值
                        a=l[0]
                        if type(tree[i][a]) == np.str_:
                            self.label_x.append(tree[i][a])
                        elif type(tree[i][a]) == dict:
                            return pre_one_label(tree[i][a])
                        else:print('tree error')
                    else:
                        a=l[1]
                        if type(tree[i][a]) == np.str_:
                            self.label_x.append(tree[i][a])
                        elif type(tree[i][a]) == dict:
                            return pre_one_label(tree[i][a])
                        else:print('tree error')
            pre_one_label(tree)
        # print(self.label_x)
        x['predict_class'] = self.label_x
        return x
    def score(self,predict_x,x):        #本算法使用F1作为评价指标
        if type(x.iloc[0,-1])==np.str_:     #分类树
            f1_l=[]
            for i in range(len(np.unique(x.iloc[:,-1]))):        #多分类任务，分别计算以一个标签为正类，其余标签为负类的得分
                positive_label = np.unique(x.iloc[:,-1])[i]
                tp = np.sum((x.iloc[:,-1]==positive_label)&(predict_x.iloc[:,-1]==positive_label))      #真正类
                fp = np.sum((x.iloc[:,-1]!=positive_label)&(predict_x.iloc[:,-1]==positive_label))      #假正类
                fn = np.sum((x.iloc[:,-1]==positive_label)&(predict_x.iloc[:,-1]!=positive_label))      #假负类
                tn = np.sum((x.iloc[:,-1]!=positive_label)&(predict_x.iloc[:,-1]!=positive_label))      #真负类
                f1 = (2*tp)/(2*tp+fp+fn)
                f1_l.append(f1)
        else:                               #回归树
            print('回归树')

        return np.mean(f1_l)




#iris数据集生成与处理
class GenerateData():
    def generate(self):
        #加载数据
        from sklearn.datasets import load_iris
        iris_data = load_iris()
        #处理数据，写入列名和标签名
        data = np.array(iris_data.data)
        target = np.array(iris_data.target)
        df = pd.DataFrame(np.c_[data,target])
        c_name = iris_data.feature_names
        c_name.append('class')
        t_name = iris_data.target_names
        df.columns = c_name
        df['class'] = df['class'].apply(lambda x:t_name[int(x)])

        return df



class GetBestModel():
    def __init__(self,k,dataframe,cart):
        '''
        :param k: 进行k次模型训练,选择在验证集上表现最好得模型
        :param tree: 传入的树模型
        :param df: 交叉验证数据
        output  :  平均得分最好的模型
        '''
        self.k = k
        self.df = df
        self.cart = cart

    def k_valid(self):
        model_list=[]           #用来储存模型得列表
        score_list=[]           #用来储存每个模型的得分的列表

        for i in range(self.k):
            self.df = self.df.sample(frac=1).reset_index(drop=True)       #打乱数据
            train_data = self.df.iloc[:90, ].reset_index(drop=True)       #训练集使用来训练模型，占比约
            # 训练模型
            tree = self.cart.create_tree(train_data)
            model_list.append(tree)

            vali_score_list=[]
            for j in range(10):
                #从验证集中随机选取70%的数据来评估模型
                validate_data = self.df.iloc[90:120,].sample(frac=0.7).reset_index(drop=True)
                #预测验证集
                pre_v = self.cart.predict(tree, validate_data.iloc[:, :-1])
                #计算验证集得分
                s1 = self.cart.score(pre_v, validate_data)
                vali_score_list.append(s1)
            #将验证集得分的平均值放入列表中
            score_list.append(np.mean(vali_score_list))
        # print(score_list)
        return model_list[np.argmax(score_list)]



#生成数据
df = GenerateData().generate()


#构建模型
cart = Cart()


#进行5次模型训练，寻找最优表现的模型
g=GetBestModel(k=5,dataframe=df,cart=cart)
cart_model = g.k_valid()

#验证模型在测试集上的表现
#生成测试集，并计算得分
test_data = df.iloc[120:, ].reset_index(drop=True)
predict= cart.predict(cart_model,test_data.iloc[:,:-1])
score = cart.score(predict,test_data)
print('the score on test data is {}'.format(score))

