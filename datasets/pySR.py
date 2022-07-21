from organic import load_data
from pysr import PySRRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate

X, y= load_data("most_used")

model = PySRRegressor(
                model_selection = "accuracy",
                binary_operators=["+", "-", "*", "/"],
                niterations=10,
                populations=10,
                topn=10)

scores = cross_validate(model, X, y,
                        cv=5, n_jobs=-1,
                        return_estimator=True)

print(scores['test_score'])
print(model)