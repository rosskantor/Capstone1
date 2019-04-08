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
from fancyimpute import SimpleFill, KNN,  IterativeSVD, IterativeImputer
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from itertools import combinations
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from keras import models, layers

class regress():
    def __init__(self):
        self.scaler = StandardScaler()

    def load_csv_2(self):
        """
        Loads modified, imputed csv data file.
        returns dataframe used by main function
        """
        self.autofinance = pd.read_csv('RossData_imputed.csv')

    def fill_blanks(self, filler=0):
        """
        dframe: name of dataframe containing blanks to be filled with zeros
        filler: values to be filled in NaN cells. Default is 0.
        returns dataframe with no blanks
        """
        self.autofinance.fillna(filler, inplace=True)

    def split_X_y_cols(self):
        """
        autofinance: dataframe name
        X_col: names of x columns to be split from autofinance DataFrame
        y_col: name of y column to be split from autofinance DataFrame
        returns X and y dataframe
        """
        self.X = self.autofinance[['Variable01', 'Variable02', 'Variable03', 'Variable04', 'Variable05','Variable07', 'Variable08','Variable09', 'Variable10', 'Variable11', \
        'Region_MT', 'Region_MW', 'Region_NE', 'Region_P', 'Region_S', 'Region_WE']]
        self.y = self.autofinance[['Result02']]

    def getdummies(self):
        """
        df: name of dataframe containing a Region column to receive dummy variables
        returns dataframe with regional dummy variables
        """
        self.autofinance=pd.get_dummies(self.autofinance,columns=['Region'], drop_first=True)
    def splitapply(self):
        """
        X: DataFrame containing X columns
        y: DataFrame containing y column
        This function creates training and test tables
        returns X and y training and test dataframes
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3)
    def vif(self):
        """
        r: DataFrame to be tested for autocorrleation
        used to test for autocorrelation
        """
        self.y_1, self.X_1 = dmatrices('Result02 ~ Variable01  +  Variable03 + Variable05 +  Variable07 +  Variable08 +  Variable09 +  Variable10 +  Variable11 +  Region_MW +  Region_NE +  Region_P + Region_S +  Region_WE', self.autofinance, return_type='dataframe')
        self.y_2, self.X_2 = dmatrices('Result02 ~ Variable03 + Variable05 +  Variable07 +  Variable08 +  Variable09 +  Variable10 +  Variable11 +  Region_MW +  Region_NE +  Region_P + Region_S +  Region_WE', self.autofinance, return_type='dataframe')
        self.vif = pd.DataFrame()
        self.vif2 = pd.DataFrame()
        self.vif["VIF Factor"] = [variance_inflation_factor(self.X_1.values, i) for i in range(self.X_1.shape[1])]
        self.vif2["VIF Factor"] = [variance_inflation_factor(self.X_2.values, i) for i in range(self.X_2.shape[1])]
        self.vif["features"] = self.X_1.columns
        self.vif2["features"] = self.X_2.columns
    def costbenefit(self):
        """
        Builds cost benefit matrix
        returns 2X2 cost benefit matrix
        """
        self.cb = np.array([[660, -120], [0, 0]])

    def standardize(self, X_train, X_test, X_col_input):
        """
        X_train: training DataFrame
        X_test: testing DataFrame
        X_col_input: name of X columns to be standardized
        returns normalized X train and test dataframes
        """
        self.scaler = StandardScaler().fit(X_train)
        self.X_train_1 = pd.DataFrame( data=scaler.transform(self.X_train), columns = X_col_input)
        self.X_test_1 = pd.DataFrame( data = scaler.transform(X_test), columns = X_col_input)

    def splitdf (self, df):
        """
        df: name of DataFrame to be imputed per KNN(5)
        """
        self.dfReg = pd.DataFrame(df.Region)
        self.df1 = pd.DataFrame(data=KNN(5).fit_transform(df.iloc[:10000].select_dtypes(exclude='object')), columns=df.iloc[:10000].select_dtypes(exclude='object').columns, index=df.iloc[:10000].select_dtypes(exclude='object').index)
        self.df2 = pd.DataFrame(data=KNN(5).fit_transform(df.iloc[10000:20000].select_dtypes(exclude='object')), columns=df.iloc[10000:20000].select_dtypes(exclude='object').columns, index=df.iloc[10000:20000].select_dtypes(exclude='object').index)
        self.df3 = pd.DataFrame(data=KNN(5).fit_transform(df.iloc[20000:].select_dtypes(exclude='object')), columns=df.iloc[20000:].select_dtypes(exclude='object').columns, index=df.iloc[20000:].select_dtypes(exclude='object').index)
        self.y = df1.append(df2)
        self.y2 = y.append(df3)
        self.df = y2.merge(dfReg, left_index=True, right_index=True)
        self.df.to_csv('RossData_imputed.csv')

    def roc_curve(self, probabilities, labels):
        '''
        INPUT: numpy array, numpy array
        OUTPUT: list, list, list

        Take a numpy array of the predicted probabilities and a numpy array of the
        true labels.
        Return the True Positive Rates, False Positive Rates, Thresholds for the
        ROC curve and the profit matrix
        '''
        self.cb = costbenefit()
        self.thresholds = np.sort(probabilities)

        self.tprs = []
        self.fprs = []
        self.profit = []

        self.num_positive_cases = labels.sum()
        self.num_negative_cases = len(labels) - self.num_positive_cases

        for self.threshold in self.thresholds:
            # With this threshold, give the prediction of each instance
            predicted_positive = probabilities >= threshold
            # Calculate the number of correctly predicted positive cases
            true_positives = np.sum(predicted_positive * labels.iloc[:, 0])
            # Calculate the number of incorrectly predicted positive cases
            false_positives = np.sum(predicted_positive) - true_positives
            # Calculate the True Positive Rate
            tpr = true_positives / float(self.num_positive_cases)
            # Calculate the False Positive Rate
            fpr = false_positives / float(self.num_negative_cases)
            # Calculate predicted negative cases
            profit.append((true_positives * cb[0][0]) +  (false_positives * cb[0][1]))
            #Populate TP

            #Populate FP

            #Populate FN

            #Populate TN

            self.fprs.append(fpr)
            self.tprs.append(tpr)

        return tprs, fprs, thresholds.tolist(), profit


    def plotter (self, y_test, X_cols, y_Cols, figname, probreturn, x_short_identifier):
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

    def model(self):
        """
        X_train: X training dataframe
        y_train: y training DataFrame
        X_test: X test dataframe
        returns list of probabilities for each test value
        """
        y = np.array(np.ravel(self.y_train)).astype(int)
        self.clf = LogisticRegressionCV(cv=5, random_state=0,multi_class='multinomial').fit(self.X_train, y)
        self.preds = self.clf.predict(self.X_test)
        self.probs = self.clf.predict_proba(self.X_test)
        self.log_accuracy = accuracy_score(self.preds, self.y_test)
        self.log_recall = recall_score(self.preds, self.y_test)
        self.log_precision = precision_score(self.preds, self.y_test)
    def model_2(self):
        """
        X_train: X training dataframe
        y_train: y training DataFrame
        X_test: X test dataframe
        returns list of probabilities for each test value
        """
        y = np.array(np.ravel(self.y_train)).astype(int)
        self.scaler.fit(self.X_train)
        self.clf = LogisticRegressionCV(cv=5, random_state=0).fit(self.scaler.transform(self.X_train), y)
        self.preds = self.clf.predict(self.scaler.transform(self.X_test))
        self.probs = self.clf.predict_proba(self.scaler.transform(self.X_test))
        self.log_accuracy = accuracy_score(self.preds, self.y_test)
        self.log_recall = recall_score(self.preds, self.y_test)
        self.log_precision = precision_score(self.preds, self.y_test)
    def tree(self):
        self.t = RandomForestClassifier(n_estimators=400, oob_score=True)
        self.t.fit(self.X_train, self.y_train.values.ravel())
        self.t_pred= self.t.predict(self.X_test)
        self.tree_accuracy = accuracy_score(self.t_pred, self.y_test.values.ravel())
        self.tree_precision = precision_score(self.t_pred, self.y_test.values.ravel())
        self.tree_recall = recall_score(self.t_pred, self.y_test.values.ravel())
    def resample(self):
        self.majority = self.autofinance[self.autofinance.Result02==0]
        self.minority = self.autofinance[self.autofinance.Result02==1]

        df_minority_upsampled = resample(self.autofinance[self.autofinance.Result02==1],
                                      replace=True,     # sample with replacement
                                      n_samples=len(self.majority),    # to match majority class
                                      random_state=123)

        self.autofinance = pd.concat([df_minority_upsampled, self.majority])
    def down_sample(self):
        self.minority_2 = self.autofinance[self.autofinance.Result02==1]
        self.majority_2 = self.autofinance[self.autofinance.Result02==0]
        self.majority_2 = self.majority_2.iloc[:len(self.minority_2)]
        self.autofinance = pd.concat([self.majority_2, self.minority_2])
    def neural_net(self):
        # Start a Neural Network
        network = models.Sequential()

        # Add fully connected layer with a Relu activation function

        network.add(layers.Dense(units=16, activation='relu', \
        input_shape=(16,)))

        #Add fully connected layer with a Relu activation function
        network.add(layers.Dense(units=16, activation='relu'))

        #Add fully connected layer with a Relu activation function
        network.add(layers.Dense(units=1, activation='sigmoid'))

        #Compile Neural Network
        network.compile(loss='binary_crossentropy', #Cross-entropy \
            optimizer='rmsprop', # Root Mean Square Propogation \
            metrics=['accuracy'] #Accuracy performance metric
            )

        history = network.fit(self.scaler.transform(self.X_train), #Features
                            self.y_train, #Target
                            epochs = 25 , #Number of iterations
                            verbose = 1, #Print Success after each epoch
                            batch_size = 100, #Number of observations per batch
                            validation_data = (self.scaler.transform(self.X_test), self.y_test)) #Test data

if __name__=="__main__":
    r = regress()
    r.load_csv_2()
    r.fill_blanks()
    r.getdummies()
    r.resample()
    r.split_X_y_cols()
    r.splitapply()
    r.model_2()
    r.tree()
    #r.neural_net()
