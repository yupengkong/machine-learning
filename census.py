import pandas as pd
import numpy as np
from chooseFeature import chooseFeature
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, scale
from sklearn.metrics import confusion_matrix, f1_score


## get the data from the whole data set and make the predictor as -1 or 1

def drop_columns(data):  # drop the some columns from original data sets
	data = data.drop(['MIGMTR1', 'MIGMTR3', 'MIGMTR4','MIGSUN', 'PEFNTVTY', 'PEMNTVTY', 'PENATVTY', 'YEAR'], axis = 1)
	return data

def drop_rows(data):  # drop few rows from original datasets
	rownum1 = np.where(data['GRINST'] == ' ?')[0]
	rownum2 = np.where(data['HHDFMX'] == ' Grandchild <18 ever marr not in subfamily')[0]
	rownum = np.concatenate((rownum1, rownum2))
	data = data.drop(data.index[rownum])
	return data

def data_transformation(data, continous, dummy, binary): # this is the important part to make sklearn load the data
                                                         # sklearn do not directly take the categorical data in string format
	le = LabelEncoder()                                  # If the feature only have binary values( such as: sex), 0,1 encoding will be fine
                                                         # if the feature has multiple levels of value, simply coding as 0,1,2,3,...
	for col1 in dummy:                                   # will not work, because you will bring additional magnitude informations between
		le.fit(data[col1])                               # different levels. Therefore, need to use OneHotEncoder() function
		data[col1] = le.transform(data[col1])
	dummydata = np.array(data[dummy])
	enc = OneHotEncoder()
	enc.fit(dummydata)
	dummydata = enc.transform(dummydata).toarray()
	#print dummydata.shape

	for col2 in binary:
		le.fit(data[col2])
		data[col2] = le.transform(data[col2])
	binarydata = np.array(data[binary])

	le.fit(data['target'])
	data['target'] = le.transform(data['target'])
	continuousdata = np.array(data[continous])
	return np.concatenate((dummydata, binarydata, continuousdata), axis = 1)


def original_data(data): 

	X = data[:, :-1]
	y = data[:, -1]
	return X, y

#-----------------------------------------------	

## the following methods try to solve the imbalance classfication by sampling

def sampling(pos_index, neg_index, size):    # random oversampling or random undersampling depends on the size of sampling  

	pos_index, neg_index = np.random.choice(pos_index, size), np.random.choice(neg_index, size)
	indices = np.concatenate((pos_index, neg_index))
	return indices

def dataNormalization(X, num):
	if num == 1:
		normX = scale(X)
	elif num == 2:
		min_max_scaler = MinMaxScaler()
		normX = min_max_scaler.fit_transform(X)
	else:
		print 'wrong parameter for data normalization'

	return normX

def decisionTree(X, y):
	clf = tree.DecisionTreeClassifier(criterion = 'gini', max_depth = 50)
	clf.fit(X, y)
	print cross_val_score(clf, X, y, cv = 5, scoring = 'f1').mean()


def randomForest(X, y):
	clf = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 33, n_jobs = -1, max_features = 50) 
	clf.fit(X,y)
	print cross_val_score(clf, X, y, cv = 5, scoring = 'accuracy').mean()

def logistic(X, y):
	clf = LogisticRegression(penalty = 'l2', C = 0.5, random_state = 33) 
	clf.fit(X,y)
	print cross_val_score(clf, X, y, cv = 5, scoring = 'accuracy').mean()

def KNN(X, y):
	clf = KNeighborsClassifier(n_neighbors = 5)
	print cross_val_score(clf, X, y, cv = 5, scoring = 'accuracy').mean()

def SVM(X, y):
	clf = SVC(C = 10)
	clf.fit(X,y)
	print cross_val_score(clf, X, y, cv = 5, scoring = 'accuracy').mean()


def chooseF(X, y):
	#print len(y)
	clf = chooseFeature()
	clf.fit(X,y)
	print cross_val_score(clf, X, y, cv = 5, scoring = 'accuracy').mean()
 



if __name__ == '__main__':

	data = pd.read_csv('census-income.data', header = None, delimiter = ',')   # read the original data 
	#data_test = pd.read_csv('census-income.test', header = None, delimiter = ',')

	data.columns = ['AAGE', 'ACLSWKR', 'ADTIND', 'ADTOCC', 'AHGA', 'AHRSPAY',
                	'AHSCOL', 'AMARITL', 'AMJIND', 'AMJOCC', 'ARACE', 'AREORGN',
                	'ASEX', 'AUNMEM', 'AUNTYPE', 'AWKSTAT', 'CAPGAIN', 'CAPLOSS',
                	'DIVVAL', 'FILESTAT', 'GRINREG', 'GRINST', 'HHDFMX', 'HHDREL', 
                	'MARSUPWT', 'MIGMTR1', 'MIGMTR3', 'MIGMTR4', 'MIGSAME', 'MIGSUN', 
                	'NOEMP','PARENT', 'PEFNTVTY', 'PEMNTVTY', 'PENATVTY', 'PRCITSHP',
                	'SEOTR','VETQVA', 'VETYN', 'WKSWORK', 'YEAR', 'target']  # original columns name from the supplementary file of data set

	continuous_columns = ['AAGE', 'AHRSPAY', 'DIVVAL', 'NOEMP', 'CAPGAIN', 'CAPLOSS', 'WKSWORK', 'MARSUPWT','target'] # names of columns with continous value( I put the 'target' also here)
	binary_columns = ['ASEX'] # columns with binary value

	data = drop_columns(data) # drop some colums
	data = drop_rows(data)  # drop some rows, since some rows have missing value

	dummy_columns = [col for col in data.columns.values if col not in continuous_columns if col not in binary_columns]# names of columns with dummy features

	data = data_transformation(data, continuous_columns, dummy_columns, binary_columns) # transform the dataset
	X, y = original_data(data) # split features and target


	pos_index, neg_index = np.where(y == 1)[0], np.where(y == 0)[0] # separate the indices of target between values of 1 and 0
	indices = sampling(pos_index, neg_index, 200000)  # correct the imbalance of the training dataset
	X, y = X[indices], y[indices] # get the final dataset for model fitting, this is for training set only, shuffling the data

	logistic(dataNormalization(X, 2), y)
	chooseF(X, y)

	randomForest(X, y)
	decisionTree(X, y)
	SVM(dataNormalization(X, 2), y) 


# 5 lines above for the test of the model, (X, y) denotes the training set, (X_test, y_test) denotes the test set
# Run these commands with the parameters we mentioned in the reports will give the F-1 score for both class