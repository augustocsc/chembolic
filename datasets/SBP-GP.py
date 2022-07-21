from organic import load_data, split_data
from pyGPGOMEA import GPGOMEARegressor as GPGR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate

X, y= load_data("all")
X_train, X_test, y_train, y_test = split_data(X, y)
model = GPGR(
                gomea=True, 
                ims='5_1', 
                generations=100, 
                parallel=16,
                functions='+_*_-_/',
                popsize=50
            )

model.fit(X_train, y_train)
fprint(model)