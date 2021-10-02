import numpy as np
import pandas as pd
import re

#cart代码
class Cart():
    def __init__(self):
        self.label_x = []       #to store predicted labels
    def gini(self,x):
        '''
        :param x: input data
        :return: Gini index of data x
        '''
        n = len(x)              #n represents the length of the dataset

        gini_D = 1 - np.sum(
            [np.square(
                np.sum(
                    x.iloc[:,-1]==np.unique(x[x.columns[x.shape[1]-1]])[i])/n)
                for i in range(len(np.unique(x.iloc[:,-1])))])
        return gini_D
    def gini_a(self, e, x,a):
        '''
        :param e:
        e is the threshold, the ones greater than e are one category, and those smaller than e are the other category
        :param x: input data
        :param a: is the column name of the input data x
        :return: Gini index of data set x under numerical feature a
        '''
        n1 = len(x)
        D1 = np.sum(x[a]<=e)
        D2 = np.sum(x[a]>e)
        gini_a = ((D1/n1)*self.gini(x[x[a]<=e])) + ((D2/n1)*self.gini(x[x[a]>e]))
        return gini_a

    # Calculate the Gini coefficient of the dataset under the character feature a,
    # e is the threshold, x is the data of the dataframe type, and a is the column name
    def gini_b(self,e,x,a):
        n2 = len(x)
        D1 = np.sum(x[a]==e)
        D2 = np.sum(x[a]!=e)
        gini_b = ((D1/n2)*self.gini(x[x[a]==e])) + ((D2/n2)*self.gini(x[x[a]!=e]))

    def SplitDataSet(self,x, a, e):
        '''
        fuction:Divide the data set into two left and right sub-data sets according to a certain value of the feature column
        :param x:Data set
        :param a:a certain characteristic column in the data set
        :param e: certain value in the characteristic column
        :return:Left and right subdataset
        '''

        matLeft = x[x[a] <= e]
        matRight = x[x[a] > e]
        return matLeft, matRight

    # Find the feature and threshold corresponding to the minimum Gini coefficient
    def search(self,x):           #x is the divided dataset, containing features and labels
        list=[0,0,1]                #Assign initial value to the list
        for i in range(x.shape[1]-1):             #for each feature
            if (type(x.iloc[0,i])== np.float64):              #feature is continuous
                c = x[x.columns[i]]                        #c is a column of data
                c = c.sort_values().values             #sort
                for j in range(len(c)-1):       #different threshold
                    e = (c[j]+c[j+1])/2           #threshold
                    try:
                        gini_i = self.gini_a(e,x,x.columns[i])      #Gini coefficient under a certain characteristic with e as the threshold
                    except:
                        print('gini_a error')
                    #Determine whether the Gini coefficient is the smallest, if it is, continue, otherwise save the new value
                    if list[2]<=gini_i:
                        continue
                    else:
                        # Make a list, the first element is the feature, the second is the threshold, and the third is the Gini coefficient
                        list = []
                        list.append(x.columns[i])
                        list.append(e)
                        list.append(gini_i)
            elif (type(x.iloc[0,i])== np.str_):                             #feature is discrete data
                for j in np.unique(x.iloc[:,i].values):
                    gini_j = self.gini_b(j,x,x.columns[i])
                    if list[2]<=gini_i:
                        continue
                    else:
                        list = []
                        list.append(x.columns[i])
                        list.append(j)
                        list.append(gini_i)


        return list[0],list[1]


    def create_tree(self,x,least_sample_number=1,least_gini=0.1):
        '''

        :param x: input data
        :param least_sample_number:
         The minimum number of samples for children node, and leaf node is generated when it is less than
        :param least_gini:
         The smallest Gini coefficient of children node, which generates leaf node when it is less than
        :return: Generate decision tree
        '''
        if (len(x)<=least_sample_number) or (len(np.unique(x.iloc[:,-1]))==1):
            return x.iloc[0,-1]               #Output label as leaf node
        a , e = self.search(x)
        matleft,matright = self.SplitDataSet(x,a,e)
        cart_tree={}
        cart_tree[a] = {}
        cart_tree[a]['<=' + str(e) + 'contains' + str(len(matleft))] = self.create_tree(matleft)
        cart_tree[a]['>' + str(e) + 'contains' + str(len(matright))] = self.create_tree(matright)
        return cart_tree
    def predict(self,tree,x):       #tree is the tree model, x is the sample of the data to be predicted
        self.label_x = []

        #Write regular, replace the threshold number in the tree
        regex = re.compile('<=|>(.+)contains')
        for j in range(len(x)):         #Predict the label of each sample separately
            def pre_one_label(tree):
                for i in tree:
                    value = x.loc[j,i]
                    l=[]
                    for a in tree[i]:
                        l.append(a)

                    epison = float(re.findall(regex,a)[0])
                    if value<=epison:       #less than threshold
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
    def score(self,predict_x,x):        #This algorithm uses F1 as an evaluation indicator
        if type(x.iloc[0,-1])==np.str_:     #decision tree
            f1_l=[]
            #For multi-classification tasks, calculate the scores of one label as the positive class
            # and the remaining labels as the negative class
            for i in range(len(np.unique(x.iloc[:,-1]))):
                positive_label = np.unique(x.iloc[:,-1])[i]
                tp = np.sum((x.iloc[:,-1]==positive_label)&(predict_x.iloc[:,-1]==positive_label))
                fp = np.sum((x.iloc[:,-1]!=positive_label)&(predict_x.iloc[:,-1]==positive_label))
                fn = np.sum((x.iloc[:,-1]==positive_label)&(predict_x.iloc[:,-1]!=positive_label))
                tn = np.sum((x.iloc[:,-1]!=positive_label)&(predict_x.iloc[:,-1]!=positive_label))
                f1 = (2*tp)/(2*tp+fp+fn)
                f1_l.append(f1)
        else:                               #Regression tree
            print('Regression tree')

        return np.mean(f1_l)




#iris Dataset generation and processing
class GenerateData():
    def generate(self):
        from sklearn.datasets import load_iris
        iris_data = load_iris()
        #Process data, write column names and label names
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
        :param k: Perform k times of model training and choose the model that performs best on the validation set
        :param tree: Inpuy tree model
        :param df: Cross-validation data
        output  :  Model with the best average score
        '''
        self.k = k
        self.df = df
        self.cart = cart

    def k_valid(self):
        model_list=[]           #The list used to store the model
        score_list=[]           #A list used to store the scores of each model

        for i in range(self.k):
            self.df = self.df.sample(frac=1).reset_index(drop=True)       #shuffle data
            train_data = self.df.iloc[:90, ].reset_index(drop=True)
            # train model
            tree = self.cart.create_tree(train_data)
            model_list.append(tree)

            vali_score_list=[]
            for j in range(10):
                # Randomly select 70% of the data from the validation set to evaluate the model
                validate_data = self.df.iloc[90:120,].sample(frac=0.7).reset_index(drop=True)
                #Prediction validation set
                pre_v = self.cart.predict(tree, validate_data.iloc[:, :-1])
                #Calculate the validation set score
                s1 = self.cart.score(pre_v, validate_data)
                vali_score_list.append(s1)
            #Put the average of the validation set scores into the list
            score_list.append(np.mean(vali_score_list))
        # print(score_list)
        return model_list[np.argmax(score_list)]



#generate data
df = GenerateData().generate()


#build model
cart = Cart()


#Perform model training 5 times to find the best performing model
g=GetBestModel(k=5,dataframe=df,cart=cart)
cart_model = g.k_valid()

#Verify the performance of the model on the test set
#Generate test set and calculate score
test_data = df.iloc[120:, ].reset_index(drop=True)
predict= cart.predict(cart_model,test_data.iloc[:,:-1])
score = cart.score(predict,test_data)
print('the score on test data is {}'.format(score))

