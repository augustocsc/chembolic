import organic as org
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import feyn

data = org.get_as_dataframe()

train, test = train_test_split(data, test_size=0.3, random_state=42)

ql = feyn.QLattice(random_seed=42)

model = ql.auto_run(
    data = train,
    output_name = "N",
    n_epochs=100
)

best = model[0]

print(best.r2_score(test))
sympy_model = best.sympify(symbolic_lr=True, include_weights=False)
print(sympy_model.as_expr())
