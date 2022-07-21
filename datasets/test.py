from organic import load_data
import numpy as np
from sklearn.model_selection import KFold
from pyGPGOMEA import GPGOMEARegressor as GPGR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from pysr import PySRRegressor

from sklearn.metrics import r2_score

X, y= load_data("all")

kf = KFold(n_splits=10, shuffle = True)

sbp = GPGR(
                gomea=True, 
                ims='5_1', 
                generations=100, 
                parallel=16,
                functions='+_*_-_/',
                popsize=50
            )

pys = PySRRegressor(
                model_selection = "accuracy",
                binary_operators=["+", "-", "*", "/"],
                niterations=10,
                populations=10,
                topn=10)

lin = LinearRegression()

results = {
    'linear' : [],
    'sbp': []
}
X = np.asarray(X)
for train_index, test_index in kf.split(X):
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test =  scaler.transform(X_test)

    scaler = StandardScaler()
    y_train = scaler.fit_transform(y_train)
    y_test =  scaler.transform(y_test)

    sbp.fit(X_train, y_train)
    #pys.fit(X_train, y_train)
    lin.fit(X_train, y_train)

    results['linear'].append(r2_score(y_test, sbp.predict(X_test)))
    results['sbp'].append(r2_score(y_test, lin.predict(X_test)))

print(results)
    