import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def explore():
    df = pd.read_csv('RossData_20181001.csv')

    df.rename(index=str, columns= {' DateOfData': 'DateOfData', ' Variable01': 'Variable01', ' Variable02': 'Variable02'
    , ' Variable03': 'Variable03',' Variable04' : 'Variable04',
    ' Variable05' : 'Variable05', ' Variable06' : 'Variable06',
    ' Variable07' : 'Variable07',' Variable08': 'Variable08', ' Variable09' : 'Variable09',
    ' Variable10' :'Variable10', ' Variable11':'Variable11',
    ' Result01': 'Result01', ' Result02' : 'Result02', ' Result03' : 'Result03'},inplace=True)

    return df
def dropper(df, columns):
    df.drop(columns=(columns),inplace=True)
    return df

def column_builder(df):
    for i, j in enumerate(df.head()):
        col = df.
        df2[]


def trainer(df, y):

    return train_test_split(df, df[y], test_size=0.20, random_state=42)

def pca(df):
    pca = PCA(.90)
    s = StandardScaler()
    df2 = s.fit_transform(df)
