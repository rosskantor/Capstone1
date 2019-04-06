import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools import add_constant
from statsmodels.discrete.discrete_model import Logit
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
       , y_col=['Result02']):
    """
    X_col: text name of X's wanted in the form of a list []
    y_col: text name of y variable in the form of a list []
    returns X_col and y_col text names
    """
    return X_col, y_col

def dict_X_cols(X=[1,3,5,7,8,10,11,'MW','S','P', 'N', 'W']):
    """
    Dictionary field accepting a list of key entries that will be translated to variable names
    Returns x and y columns fed into regression model
    """
    X_col={1:'Variable01', 3:'Variable03', 5:'Variable05', 7:'Variable07', 8:'Variable08',
    9:'Variable09', 10:'Variable10', 11:'Variable11', 'MW':'Region_MW', 'MT':'Region_MT',
    'N':'Region_NE', 'P':'Region_P', 'S':'Region_S', 'W':'Region_WE'}
    y_col=['Result02']
    X_col_input = [X_col[i] for i in X]
    y_col_input = y_col
    return X_col_input, y_col_input


def load_csv():
    """
    Loads raw csv, shrinks variable names, adds region lookup table and drops column names
    returns dataframe
    """
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
    """
    Loads modified, imputed csv data file.
    returns dataframe used by main function
    """
    autofinances = pd.read_csv('RossData_imputed.csv')
    return autofinances

def fill_blanks(dframe, filler=0):
    """
    dframe: name of dataframe containing blanks to be filled with zeros
    filler: values to be filled in NaN cells. Default is 0.
    returns dataframe with no blanks
    """
    dframe.fillna(filler, inplace=True)
    return dframe
def split_X_y_cols(autofinance, X_col, y_col):
    """
    autofinance: dataframe name
    X_col: names of x columns to be split from autofinance DataFrame
    y_col: name of y column to be split from autofinance DataFrame
    returns X and y dataframe
    """
    X = autofinance.ix[:, X_col]
    y = autofinance.ix[:, y_col]
    return X, y
def getdummies(df):
    """
    df: name of dataframe containing a Region column to receive dummy variables
    returns dataframe with regional dummy variables
    """
    afdummies=pd.get_dummies(df,columns=['Region'])
    return afdummies
def splitapply(X, y):
    """
    X: DataFrame containing X columns
    y: DataFrame containing y column
    This function creates training and test tables
    returns X and y training and test dataframes
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    return X_train, X_test, y_train, y_test
def vif(r):
    """
    r: DataFrame to be tested for autocorrleation
    used to test for autocorrelation
    """
    y, X = dmatrices('Result02 ~ Variable01  +  Variable03 + Variable05 +  Variable07 +  Variable08 +  Variable09 +  Variable10 +  Variable11 +  Region_MW +  Region_NE +  Region_P + Region_S +  Region_WE', r, return_type='dataframe')
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["features"] = X.columns
def costbenefit():
    """
    Builds cost benefit matrix
    returns 2X2 cost benefit matrix
    """
    cb = np.array([[660, -120], [0, 0]])
    return cb

def standardize(X_train, X_test, X_col_input):
    """
    X_train: training DataFrame
    X_test: testing DataFrame
    X_col_input: name of X columns to be standardized
    returns normalized X train and test dataframes
    """
    scaler = StandardScaler().fit(X_train)
    X_train_1 = pd.DataFrame( data=scaler.transform(X_train), columns = X_col_input)
    X_test_1 = pd.DataFrame( data = scaler.transform(X_test), columns = X_col_input)

    return X_train_1, X_test_1

def splitdf (df):
    """
    df: name of DataFrame to be imputed per KNN(5)
    """
    dfReg = pd.DataFrame(df.Region)
    df1 = pd.DataFrame(data=KNN(5).fit_transform(df.iloc[:10000].select_dtypes(exclude='object')), columns=df.iloc[:10000].select_dtypes(exclude='object').columns, index=df.iloc[:10000].select_dtypes(exclude='object').index)
    df2 = pd.DataFrame(data=KNN(5).fit_transform(df.iloc[10000:20000].select_dtypes(exclude='object')), columns=df.iloc[10000:20000].select_dtypes(exclude='object').columns, index=df.iloc[10000:20000].select_dtypes(exclude='object').index)
    df3 = pd.DataFrame(data=KNN(5).fit_transform(df.iloc[20000:].select_dtypes(exclude='object')), columns=df.iloc[20000:].select_dtypes(exclude='object').columns, index=df.iloc[20000:].select_dtypes(exclude='object').index)
    y = df1.append(df2)
    y2 = y.append(df3)
    df = y2.merge(dfReg, left_index=True, right_index=True)
    df.to_csv('RossData_imputed.csv')

def roc_curve(probabilities, labels):
    '''
    INPUT: numpy array, numpy array
    OUTPUT: list, list, list

    Take a numpy array of the predicted probabilities and a numpy array of the
    true labels.
    Return the True Positive Rates, False Positive Rates, Thresholds for the
    ROC curve and the profit matrix
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


def plotter (y_test, X_cols, y_Cols, figname, probreturn, x_short_identifier):
    """
    y_test: y test dataframe
    X_cols: name of x columns
    y_Cols: name of y column
    figname: name of plot (probably used)
    probreturn: list of probabilities to be passed to grapher
    x_short_identifier: keys identifying X column short names
    returns profit matrix and graph
    """
    tpr, fpr, thresholds , profit= roc_curve(probreturn, y_test)

    fig, ax1 = plt.subplots(figsize=(13, 6))
    plt.rcParams.update({'font.size': 18})
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
    plt.title("ROC and Profitability plot of features \n " + str(x_short_identifier) + " Max Profit " + str(max(profit)))
    plt.savefig(figname + '.png')
    plt.show()
    return  profit

def model(X_train, y_train, X_test):
    """
    X_train: X training dataframe
    y_train: y training DataFrame
    X_test: X test dataframe
    returns list of probabilities for each test value
    """
    y = np.array(np.ravel(y_train)).astype(int)
    clf = LogisticRegressionCV(cv=5, random_state=0,multi_class='multinomial').fit(X_train, y)
    probs = clf.predict_proba(X_test)
    return probs[:, 1]
def confusionmatrix(df, probs, threshold=[0.15, 0.25, 0.45, 0.70]):
    """
    df: y test DataFrame
    probs: probabilities for success
    threshold: cut of points used to measure TP and FP rates
    returns confusion matrix
    """
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

def regressionoutput(X_train, y_train):
    """
    Does not work as laid out.  Should return logit regression summary.
    """
    X = X_train
    X_const = add_constant(X, prepend=True)
    y = y_train
    logit_model = Logit(y, X_const).fit()
    print(logit_model.summary())
def main(x_short_identifier,figname='Default'):
    """
    x_short_identifier: keys identifiers in list [] format pointing to X col names
    figname: name of plot figure name defaulted to 'Defualt'
    returns confusion matrix
    """
    X_col_input, y_col_input = dict_X_cols(x_short_identifier)
    cb = costbenefit()
    autofinance = load_csv_2()
    df = fill_blanks(autofinance)
    af_mod = getdummies(df)
    X_col, y_col = X_y_columns(X_col = X_col_input, y_col= y_col_input)
    X, y = split_X_y_cols(af_mod, X_col, y_col)
    X_train, X_test, y_train, y_test = splitapply(X, y)
    X_train_1, X_test_1=standardize(X_train, X_test, X_col_input)
    probreturn = model(X_train_1, y_train, X_test_1)
    plotter(y_test, X_col, y_col, figname, probreturn, x_short_identifier)
    cfmatrix = confusionmatrix(y_test, probreturn)
    return cfmatrix
