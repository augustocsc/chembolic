from organic import load_data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate

X, y= load_data("all")
reg = LinearRegression()

scores = cross_validate(reg, X, y,
                        cv=5, n_jobs=-1,
                        scoring = 'r2',
                        return_estimator=True)

print(scores['test_score'])
print(reg)
