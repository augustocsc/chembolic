from organic import load_data
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from pstree.cluster_gp_sklearn import PSTreeRegressor, GPRegressor

X_train, X_test, y_train, y_test= load_data("all")

model = PSTreeRegressor(regr_class=GPRegressor, tree_class=DecisionTreeRegressor,
                         height_limit=6, n_pop=20, n_gen=5,
                         adaptive_tree=True, basic_primitive='optimal', size_objective=True)

model.fit(X_train, y_train)

print(model.model())
print(r2_score(y_test, model.predict(X_test)))
