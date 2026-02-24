import pandas as pd
import mlflow
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("iris_data.csv")
X = data.drop("target", axis=1)
y = data["target"]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

param_grid = {"model__C": [0.1, 1, 10]}
grid = GridSearchCV(pipeline, param_grid, cv=2)

mlflow.start_run()
grid.fit(Xtrain, ytrain)
best_model = grid.best_estimator_
accuracy = best_model.score(Xtest, ytest)

mlflow.log_param("C", grid.best_params_["model__C"])
mlflow.log_metric("accuracy", accuracy)

joblib.dump(best_model, "model.pkl")
mlflow.end_run()

print("Accuracy:", accuracy)
