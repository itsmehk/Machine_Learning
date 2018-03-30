import pandas as pd
import numpy as np
from sklearn import svm, ensemble, cross_validation, preprocessing
import pdb
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.cross_validation import KFold

def mcrmse(y_predicted, y_true, num_crossval, num_predict):
    total = 0.
    for i in xrange(num_predict):
        inner_total = 0.
        for j in xrange(num_crossval):
             inner_total = inner_total + (y_true[j,i] - \
                 y_predicted[j, i])**2
        total = total + (1./num_crossval * inner_total)**0.5
    metric = 1./num_predict * total
    return metric


def rmse(y_predicted, y_true, num_crossval):
    """
    Returns floating point value of the RMSE for a given column.
    """
    total = 0.
    for j in range(num_crossval):
        total = total + (y_true[j] - y_predicted[j])**2
    metric = ( total / num_crossval )**0.5
    return metric


filter_width = 3
interp_size = 3500
interp_size_sand = 3500

train=pd.read_csv('/home/himank/Machine Learning/Africa_Soil/training.csv')
test = pd.read_csv('/home/himank/Machine Learning/Africa_Soil/sorted_test.csv')
train.head()


labels = train[['Ca','P','pH','SOC','Sand']].values

train.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN'], axis=1, inplace=True)
test.drop('PIDN', axis=1, inplace=True)
predictedvals = ['Ca', 'P', 'pH', 'SOC', 'Sand']

# Defining spectrum X-axis
min_k = 599.76
max_k = 7497.96
knew = np.linspace(max_k, min_k, num=interp_size)
knewsand = np.linspace(max_k, min_k, num=interp_size_sand)
kold = np.linspace(max_k, min_k, num=3578)
xnew = np.linspace(0, 3578-1, num=interp_size)
xnewsand = np.linspace(0, 3578-1, num=interp_size_sand)
xold = np.linspace(0, 3578-1, num=3578)

# Defining matrices for training and testing sets
xtrainraw, xtestraw = np.array(train), np.array(test)
xtrain = np.zeros((xtrainraw.shape[0],
                       interp_size + xtrainraw[:, 3578:].shape[1]))
xtest = np.zeros((xtestraw.shape[0],
                       interp_size + xtestraw[:, 3578:].shape[1]))
xtrainsand = np.zeros((xtrainraw.shape[0],
                       interp_size_sand + xtrainraw[:, 3578:].shape[1]))
xtestsand = np.zeros((xtestraw.shape[0],
interp_size_sand + xtestraw[:, 3578:].shape[1]))

# K-fold cross-validation for determining model accuracy
crossval = KFold(len(xtrain), n_folds=15)

# Model Training: Use support vector regressor

ca_vec = svm.SVR(C=10000.0)
p_vec = svm.SVR(C=10000.0)
ph_vec = svm.SVR(C=1000.0)
soc_vec = svm.SVR(C=6000.0)
sand_vec = svm.SVR(C=800.0)

preds = np.zeros((xtest.shape[0], 5))
trainpreds = np.zeros((xtrain.shape[0], 5))

from sklearn import datasets, linear_model
regr = linear_model.LinearRegression()
for i in range(5):

        resultstr = []
        resultscv = []
        # Select the K best features to use for training
        featsel = SelectKBest(f_regression, k=3505)
        models = [ca_vec, p_vec, ph_vec, soc_vec, sand_vec]
        # Feature scaling: Adjust mean to 0 and stdev to 1
        selxtrain = featsel.fit_transform(xtrain, labels[:, i])
        selxtest = featsel.transform(xtest)

        for traincv, testcv in crossval:
            selxtr = selxtrain[traincv]
            selytr = labels[traincv, i]
            selxte = selxtrain[testcv]
            selyte = labels[testcv, i]

            models = [ca_vec, p_vec, ph_vec, soc_vec, sand_vec]
            # FIT MODELS
            regr.fit(selxtr, selytr)
            mytrainpreds = regr.predict(selxtr).astype(float)
            cvpreds = regr.predict(selxte).astype(float)

            resultstr.append(rmse(mytrainpreds, selytr, mytrainpreds.shape[0]))
            resultscv.append(rmse(cvpreds, selyte, cvpreds.shape[0]))

        print ('Train results for ', predictedvals[i] +\
                              str(np.array(resultstr).mean()))
        print ('CV results for ', predictedvals[i] +\
                              str(np.array(resultscv).mean()))

        preds[:,i] = regr.predict(selxtest).astype(float)


sample = pd.read_csv('sample_submission.csv')
sample['Ca'] = preds[:,0]
sample['P'] = preds[:,1]
sample['pH'] = preds[:,2]
sample['SOC'] = preds[:,3]
sample['Sand'] = preds[:,4]

sample.to_csv('redshiftzero_predictions.csv', index = False)