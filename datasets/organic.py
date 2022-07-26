from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import make_pipeline

#Rice and Beans
import pandas as pd
import numpy as np

columns = []
features = {'pearson': ['e+LUMO', 'EPA', 'μ+', 'eHOMO', 'e+HOMO', 'μ', 'SNu', 'q', 'SH'],
            'spearman': ['sN', 'μ+', 'eHOMO', 'μ', 'e+HOMO', 'q', 'SNu', 'MK'],
            'most_used': ['sN', 'B5', 'SH', 'SInt', 'EPA', 'SNu', 'e+LUMO', 'eHOMO', 'ε', 'η'],
            'base': ['ε', 'q', 'eHOMO', '%VH', 'B1', 'EPA', 'SNu', 'SInt', 'SH']}

def get_data(select_columns = 'all'):
  #Reading the data
  data = get_as_dataframe(select_columns)

  if select_columns == 'all':
    X = data.drop(['N'], axis = 1)
  else:
    X = data[features[select_columns]]
  
  columns = data.columns
  
  y = np.asarray(data['N'])
  y = y.reshape(-1, 1)
  
  return X, y

def get_as_dataframe(select_columns = 'all'):
  df = pd.read_csv('data/organicData.txt', delim_whitespace=True)
  df = df.drop(['Nu', 'solvent'], axis = 1)
  if select_columns != 'all':
    df = df[features[select_columns]]
  return df

def get_splited_data(test_size = .3, select_columns = 'all'):
    
    X, y = get_data(select_columns)

    if test_size > 0:
      #Separing train/test
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42)

      scaler = StandardScaler()
      X_train = scaler.fit_transform(X_train)
      X_test =  scaler.transform(X_test)

      scaler = StandardScaler()
      y_train = scaler.fit_transform(y_train)
      y_test =  scaler.transform(y_test)
      
      return X_train, X_test, y_train, y_test