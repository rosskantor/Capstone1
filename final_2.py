import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from fancyimpute import KNN
from sklearn.linear_model import LogisticRegressionCV
from sklearn.decomposition import PCA
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from keras import models, layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
class regress():
    def __init__(self):
        self.scaler = StandardScaler()

    def load_csv_2(self):
        """
        Loads modified, imputed csv data file.
        returns dataframe used by main function
        """
        self.autofinance = pd.read_csv('../data/RossData_imputed.csv')

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

    def model(self):
        """
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
        """
        Train, predict and measure randomforest classifier
        """
        self.t = RandomForestClassifier(n_estimators=400, oob_score=True)
        self.t.fit(self.X_train, self.y_train.values.ravel())
        self.t_pred= self.t.predict(self.X_test)
        self.tree_accuracy = accuracy_score(self.t_pred, self.y_test.values.ravel())
        self.tree_precision = precision_score(self.t_pred, self.y_test.values.ravel())
        self.tree_recall = recall_score(self.t_pred, self.y_test.values.ravel())
    def model_selector(self):
        """
        Academic attempt at building a pipeline interfacing with a gridsearch object.
        """
        np.random.seed(0)
        preprocess = FeatureUnion([('std', StandardScaler()), ('pca', PCA())])
        pipe = Pipeline([('preprocess', preprocess),
                        ('classifier', LogisticRegression())])
        search_space = [{'preprocess__pca__n_components': [1,2,3,4,5,6,7,8,9,10],
                        'classifier__penalty': ['l1', 'l2'],
                        'classifier__C': np.logspace(0 , 4, 10)}]
        clf = GridSearchCV(pipe, search_space, cv=5, verbose=0, n_jobs=-1)
        clf.fit(self.X, self.y)
    def resample(self):
        """
        Upsample minority class to the same length as the majority class. Does not include random samples.
        """
        self.majority = self.autofinance[self.autofinance.Result02==0]
        self.minority = self.autofinance[self.autofinance.Result02==1]

        df_minority_upsampled = resample(self.autofinance[self.autofinance.Result02==1],
                                      replace=True,     # sample with replacement
                                      n_samples=len(self.majority),    # to match majority class
                                      random_state=123)

        self.autofinance = pd.concat([df_minority_upsampled, self.majority])
    def down_sample(self):
        """
        Downsample majority class to the same length as the majority class.  Does not include random samples.
        """
        self.minority_2 = self.autofinance[self.autofinance.Result02==1]
        self.majority_2 = self.autofinance[self.autofinance.Result02==0]
        self.majority_2 = self.majority_2.iloc[:len(self.minority_2)]
        self.autofinance = pd.concat([self.majority_2, self.minority_2])
    def neural_net(self):
        """
        Train a basic neural net model.
        """
        # Start a Neural Network
        network = models.Sequential()

        # Add fully connected layer with a Relu activation function

        network.add(layers.Dense(units=300, activation='relu', input_shape=(16,)))

        #Add fully connected layer with a Relu activation function
        network.add(layers.Dense(units=300, activation='relu'))

        #Add fully connected layer with a Relu activation function
        network.add(layers.Dense(units=1, activation='sigmoid'))

        #Compile Neural Network
        network.compile(loss='binary_crossentropy', #Cross-entropy \
            optimizer='rmsprop', # Root Mean Square Propogation \
            metrics=['accuracy'] #Accuracy performance metric
            )
        """
        callbacks = [EarlyStopping(monitor='val_loss', patience=4),
                        ModelCheckpoint(filepath='best_model.h5',
                                        monitor='val_loss',
                                        save_best_only=True)]

        history = network.fit(self.scaler.transform(self.X_train), #Features
                            self.y_train, #Target
                            epochs = 25 , #Number of iterations
                            verbose = 1, #Print Success after each epoch
                            batch_size = 100, #Number of observations per batch
                            callbacks=callbacks, #Adding early stop and save best model
                            validation_data = (self.scaler.transform(self.X_test), self.y_test)) #Test data

        self.nn_preds = network.predict(self.scaler.transform(self.X_test))
        self.nn_probs = network.predict_proba(self.scaler.transform(self.X_test))

        print('The accuracy of the model is ' + str(accuracy_score(np.where(self.nn_preds>=0.50,1,0),self.y_test)))"""

        kc = KerasClassifier(build_fn=network, verbose=0)

        epochs = [5,10]
        batches = [5,10,100]
        optimizers = ['rmsprop', 'adam']

        hyperparameters = dict(optimizer=optimizers, epochs=epochs, batch_size=batches)

        grid = GridSearchCV(estimator=kc, param_grid=hyperparameters)

        self.grid_result = grid.fit(self.scaler.transform(self.X_train), self.y_train)

if __name__=="__main__":
    r = regress()
    r.load_csv_2()
    r.fill_blanks()
    r.getdummies()
    r.down_sample()
    r.split_X_y_cols()
    #r.model_selector()
