import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import seaborn as sns
import numpy as np
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import het_goldfeldquandt
import itertools as it

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import missingno as msno
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from fancyimpute import SimpleFill, KNN,  IterativeSVD, IterativeImputer
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from itertools import combinations
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor

def X_y_columns(X_col=['Variable01', 'Variable03', 'Variable05', 'Variable07', 'Variable08',
       'Variable09', 'Variable10', 'Variable11', 'Region_MW',
       'Region_NE', 'Region_P', 'Region_S', 'Region_WE']
       , y_col=['Result01']):
        return X_col, y_col
def load_csv():
            autofinances = pd.read_csv('RossData_20181001.csv')

            autofinances.rename(index=str, columns= {' DateOfData': 'DateOfData', ' Variable01': 'Variable01', ' Variable02': 'Variable02'
            , ' Variable03': 'Variable03',' Variable04' : 'Variable04',
            ' Variable05' : 'Variable05', ' Variable06' : 'Variable06',
            ' Variable07' : 'Variable07',' Variable08': 'Variable08', ' Variable09' : 'Variable09',
            ' Variable10' :'Variable10', ' Variable11':'Variable11',
            ' Result01': 'Result01', ' Result02' : 'Result02', ' Result03' : 'Result03'},inplace=True)

            s = pd.read_csv('StateLookup.csv')

            autofinance = autofinances.merge(s, how = 'left',left_on = 'Variable06', right_on='State')

            autofinance.drop(['State', 'count', 'Variable06'], axis=1, inplace=True)

            return autofinance
def load_csv_2():
    autofinances = pd.read_csv('RossData_imputed.csv')
    return autofinances

def fill_blanks(dframe, filler=0):
    dframe.fillna(filler, inplace=True)
    return dframe
def split_X_y_cols(autofinance, X_col, y_col):
    X = autofinance.ix[:, X_col]
    y = autofinance.ix[:, y_col]
    return X, y
def getdummies(df):
    afdummies=pd.get_dummies(df,columns=['Region'])
    return afdummies
def splitapply(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    return X_train, X_test, y_train, y_test
def vif():
    y, X = dmatrices('Result02 ~ Variable01  +  Variable03 + Variable05 +  Variable07 +  Variable08 +  Variable09 +  Variable10 +  Variable11 +  Region_MW +  Region_NE +  Region_P + Region_S +  Region_WE', r, return_type='dataframe')
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["features"] = X.columns
def costbenefit():
    cb = np.array([[660, -120], [0, 0]])
    return cb
def k_fold_linear(X_train, y_train, X_col):
    ''' Returns error for k-fold cross validation. '''
    err_linear, index, num_folds = 0, 0, 10
    kf = KFold(n_splits= num_folds)
    error = []
    probabilities = []
    logistic = LogisticRegression()
    y = np.array(np.ravel(y_train)).astype(int)
    for train, test in kf.split(X_train):
        logistic.fit(X_train.iloc[train], y_train.iloc[train])
        pred = logistic.predict(X_train.iloc[test])
        probabilities.append(logistic.predict_proba(X_train.iloc[test]))
        #r2 = r2_score(y, pred, multioutput='variance_weighted')
        #adj_r2(X_train, len(X_col), r2)
        error.append(rmse(pred, y[test]))

    return np.mean(error), probabilities

def rmse(theta, thetahat):
	''' Compute Root-mean-squared-error '''
	return np.sqrt(np.mean((theta - thetahat) ** 2))

def standardize(X_train, X_test, X_col_input):
    scaler = StandardScaler().fit(X_train)
    X_train_1 = pd.DataFrame( data=scaler.transform(X_train), columns = X_col_input)
    X_test_1 = pd.DataFrame( data = scaler.transform(X_test), columns = X_col_input)

    return X_train_1, X_test_1

def impute_df(df, algorithm, cols):
    """Returns completed dataframe given an imputation algorithm"""
    return pd.DataFrame(data=algorithm.fit_transform(df), columns=cols, index=df.index)

def main(X_col_input, y_col_input):
    autofinance = load_csv()

def determine_impute(df):
    """Iterates various imputation methods to find lower MSE"""
    algorithms = [SimpleFill(), KNN(1), KNN(2), KNN(3), KNN(
        4), KNN(5), IterativeSVD(), IterativeImputer()]
    RMSE_dict = {}
    df_incomplete = create_test_df(df, 0.1)
def splitdf (df):
    dfReg = pd.DataFrame(df.Region)
    df1 = pd.DataFrame(data=KNN(5).fit_transform(df.iloc[:10000].select_dtypes(exclude='object')), columns=df.iloc[:10000].select_dtypes(exclude='object').columns, index=df.iloc[:10000].select_dtypes(exclude='object').index)
    df2 = pd.DataFrame(data=KNN(5).fit_transform(df.iloc[10000:20000].select_dtypes(exclude='object')), columns=df.iloc[10000:20000].select_dtypes(exclude='object').columns, index=df.iloc[10000:20000].select_dtypes(exclude='object').index)
    df3 = pd.DataFrame(data=KNN(5).fit_transform(df.iloc[20000:].select_dtypes(exclude='object')), columns=df.iloc[20000:].select_dtypes(exclude='object').columns, index=df.iloc[20000:].select_dtypes(exclude='object').index)
    y = df1.append(df2)
    y2 = y.append(df3)
    df = y2.merge(dfReg, left_index=True, right_index=True)
    df.to_csv('RossData_imputed.csv')
def missingdata(df):
    a = msno.matrix(df, figsize=(16,7), fontsize=(8))
    a.plot()
    plt.savefig('MissingData' + '.png')
    plt.show()
def roc_curve(probabilities, labels):
    '''
    INPUT: numpy array, numpy array
    OUTPUT: list, list, list

    Take a numpy array of the predicted probabilities and a numpy array of the
    true labels.
    Return the True Positive Rates, False Positive Rates and Thresholds for the
    ROC curve.
    '''
    cb = costbenefit()
    thresholds = np.sort(probabilities)

    tprs = []
    fprs = []
    profit = []

    num_positive_cases = labels.sum()
    num_negative_cases = len(labels) - num_positive_cases

    for threshold in thresholds:
        # With this threshold, give the prediction of each instance
        predicted_positive = probabilities >= threshold
        # Calculate the number of correctly predicted positive cases
        true_positives = np.sum(predicted_positive * labels.iloc[:, 0])
        # Calculate the number of incorrectly predicted positive cases
        false_positives = np.sum(predicted_positive) - true_positives
        # Calculate the True Positive Rate
        tpr = true_positives / float(num_positive_cases)
        # Calculate the False Positive Rate
        fpr = false_positives / float(num_negative_cases)
        # Calculate predicted negative cases
        profit.append((true_positives * cb[0][0]) +  (false_positives * cb[0][1]))
        #Populate TP

        #Populate FP

        #Populate FN

        #Populate TN

        fprs.append(fpr)
        tprs.append(tpr)

    return tprs, fprs, thresholds.tolist(), profit

def calculate_probabilities(X_train_1, y_train, X_test_1):

    y = np.array(np.ravel(y_train)).astype(int)
    clf = LogisticRegressionCV(cv=5, random_state=0,multi_class='multinomial').fit(X_train_1, y)
    probs = clf.predict_proba(X_test_1)
    predict = clf.predict(X_test_1)
    coefs = clf.coef_
    probreturn = probs[:,1]
    return coefs, probreturn

def plotter (y_test, X_cols, y_Cols, figname, probreturn):
    tpr, fpr, thresholds , profit= roc_curve(probreturn, y_test)

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax1.plot(fpr, tpr, 'b')
    ax2.plot(thresholds, profit, 'r')
    x = np.linspace(0, 1, len(thresholds))
    ax3.plot(x,x, linestyle='--')
    ax1.set_xlabel("False Positive Rate (1 - Specificity)")
    ax1.set_ylabel("True Positive Rate (Sensitivity, Recall)", color = 'b')
    ax2.set_ylabel("Profit", color='r')
    ax3.set_yticklabels([])
    plt.title("ROC plot of " + str(X_cols))
    plt.savefig(figname + '.png')
    plt.show()
    return  profit

def calculate_probabilities_2(X_train, X_test, y_train, y_test, X_cols, y_Cols, figname='Nothing'):

    y = np.array(np.ravel(y_train)).astype(int)
    clf = LogisticRegressionCV(cv=5, random_state=0,multi_class='multinomial').fit(X_train, y)
    probs = clf.predict_proba(X_train)

    tpr, fpr, thresholds , profit= roc_curve(probs[:,1], y_train)

    return profit, X_cols
def adj_r2(X, predictors, r2):
	sampleSize = len(X)
	num = (1-r2)*(sampleSize - 1)
	den = sampleSize - predictors - 1
	return 1 - (num/den)

def generator(X_cols):
    stop = len(X_cols)
    A = []
    for i in range(1,stop):
        A.append(list(combinations(X_cols, i)))
    return A

def stackdf (df):

    for i, alg in enumerate(algorithms):
        X_complete = impute_df(df_incomplete, alg)
        alg_rmse = np.sqrt(mean_squared_error(df, X_complete))
        RMSE_dict[str(i)+"_"+alg.__class__.__name__] = alg_rmse # This is a fancy trick to get the class names!
    return df.from_dict(RMSE_dict, orient='index', columns=['RMSE']).sort_values(by='RMSE') # Return sorted DF

def model(X_train, y_train, X_test):
    y = np.array(np.ravel(y_train)).astype(int)
    clf = LogisticRegressionCV(cv=5, random_state=0,multi_class='multinomial').fit(X_train, y)
    probs = clf.predict_proba(X_test)
    return probs[:, 1]
def confusionmatrix(df, probs, threshold=[0.15, 0.25, 0.45, 0.70]):
    df['Probs'] = probs
    cfmatrix = []
    for i in range(len(threshold)):
        dfIn = df[df['Probs'] > threshold[i]]
        dfOut = df[df['Probs'] <= threshold[i]]

        TP = int(dfIn.iloc[:,0].sum())
        FP = int(len(dfIn) - TP)

        FN = int(dfOut.iloc[:,0].sum())
        TF = int(len(dfOut) - FN)
        tup = TP, FP, FN, TF
        cfmatrix.append(tup)

        del dfIn
        del dfOut
    return cfmatrix
def main(X_col_input,figname):
    #autofinance = load_csv()
    y_col_input = ['Result02']
    X_col_input = ['Variable01','Variable03','Variable05','Variable07','Variable08','Variable09','Variable10','Variable11','Region_MW','Region_NE','Region_P','Region_S','Region_WE']
    cb = costbenefit()
    autofinance = load_csv_2()
    df = fill_blanks(autofinance)
    af_mod = getdummies(df)
    X_col, y_col = X_y_columns(X_col = X_col_input, y_col= y_col_input)
    X, y = split_X_y_cols(af_mod, X_col, y_col)
    X_train, X_test, y_train, y_test = splitapply(X, y)
    X_train_1, X_test_1=standardize(X_train, X_test, X_col_input)
    #err, probs = k_fold_linear(X_train_1, y_train, X_col)
    probreturn = model(X_train, y_train, X_test)
    plotter(y_test, X_col, y_col, figname, probreturn)
    cfmatrix = confusionmatrix(y_test, probreturn)
    return cfmatrix

    """pd.DataFrame(data=KNN(5).fit_transform(df.iloc[:10000].select_dtypes(exclude='object')), columns=df.iloc[:10000].select_dtypes(exclude='object').columns, index=df.iloc[:10000].select_dtypes(exclude='object').index)"""

    """'Variable01, Variable02, Variable03, Variable04,
       Variable05, Variable07, Variable08, Variable09,
       Variable10, Variable11,
       Region_MT, Region_MW, Region_NE, Region_P, Region_PL,
       Region_S, Region_WE'

        y, X = dmatrices('Result02 ~ Variable01  +  Variable03 + Variable05 +  Variable07 +  Variable08 +  Variable09 +  Variable10 +  Variable11 +  Region_MW +  Region_NE +  Region_P +  Region_PL + Region_S +  Region_WE', r, return_type='dataframe')

        y, X = dmatrices('Result02 ~ Variable01, Variable02', r, return_type='dataframe'

        results1 = smf.logit('Result02 ~ Variable01  +  Variable03 + Variable05 +  Variable07 +  Variable08 +  Variable09 +  Variable10 +  Variable11 +  Region_MW +  Region_NE +  Region_P +  Region_PL + Region_S +  Region_WE', data=r).fit()
       """
