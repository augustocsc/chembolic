from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import make_pipeline

#Rice and Beans
import pandas as pd
import numpy as np

features = {'pearson': ['e+LUMO', 'EPA', 'μ+', 'eHOMO', 'e+HOMO', 'μ', 'SNu', 'q', 'SH'],
            'spearman': ['sN', 'μ+', 'eHOMO', 'μ', 'e+HOMO', 'q', 'SNu', 'MK'],
            'most_used': ['sN', 'B5', 'SH', 'SInt', 'EPA', 'SNu', 'e+LUMO', 'eHOMO', 'ε', 'η'],
            'base': ['ε', 'q', 'eHOMO', '%VH', 'B1', 'EPA', 'SNu', 'SInt', 'SH']}

def get_data(columns = 'all',  test_size):
  #Reading the data
  data = pd.read_csv('data/organicData.txt', delim_whitespace=True)

  #Removing Categorical columns
  data = data.drop(['Nu', 'solvent'], axis = 1)
  if columns == 'all':
    X = data.drop(['N'], axis = 1)
  else:
    X = data[features[columns]]
  
  y = np.asarray(data['N'])
  y = yf.reshape(-1, 1)
  
  if test_size:

    #Separing train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test =  scaler.transform(X_test)

    scaler = StandardScaler()
    y_train = scaler.fit_transform(y_train)
    y_test =  scaler.transform(y_test)
    
    return X_train, X_test, y_train, y_test
  
  return X, y
