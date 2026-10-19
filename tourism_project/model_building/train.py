import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import xgboost as xgb
import joblib
import mlflow

# 0. Ensure target folders exist
os.makedirs("tourism_project/model_building", exist_ok=True)
os.makedirs("tourism_project/deployment", exist_ok=True)

# 1. Set MLflow tracking URI and experiment name
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Tourism_Package_Prediction")

# 2. Load data safely (Checks workflow artifacts first, falls back to raw CSV if needed)
if os.path.exists("Xtrain.csv") and os.path.exists("Xtest.csv"):
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze()
    ytest = pd.read_csv("ytest.csv").squeeze()
elif os.path.exists("tourism_project/data/tourism.csv"):
    df = pd.read_csv("tourism_project/data/tourism.csv")
    df = df.drop(columns=["Unnamed: 0", "CustomerID"], errors="ignore")
    df['Gender'] = df['Gender'].replace({'fe male': 'Female'})
    df['MaritalStatus'] = df['MaritalStatus'].replace({'Unmarried': 'Single'})
    
    X = df.drop(columns=["ProdTaken"])
    y = df["ProdTaken"]
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
else:
    raise FileNotFoundError("Dataset 'tourism.csv' or train/test CSV artifacts could not be found.")

# 3. Segregate numerical and categorical features
numeric_features = [
    'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting', 
    'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 
    'Passport', 'PitchSatisfactionScore', 'OwnCar', 
    'NumberOfChildrenVisiting', 'MonthlyIncome'
]

categorical_features = [
    'TypeofContact', 'Occupation', 'Gender', 
    'ProductPitched', 'MaritalStatus', 'Designation'
]

# 4. Calculate class weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# 5. Define preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# 6. Define base XGBoost model with class weight
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric='logloss'
)

# 7. Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# 8. Define hyperparameter grid
param_grid = {
    "xgbclassifier__n_estimators": [50, 100],
    "xgbclassifier__max_depth": [2, 3],
    "xgbclassifier__learning_rate": [0.05, 0.1]
}

# 9. Start MLflow run & Hyperparameter tuning
with mlflow.start_run(run_name="Tourism_Production_XGB"):
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring="recall", n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]
        std_score = results["std_test_score"][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    mlflow.log_params(grid_search.best_params_)

    # Evaluate best model
    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1-score": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1-score": test_report["1"]["f1-score"]
    })

    print(classification_report(ytest, y_pred_test))

    # 10. Save model and log as MLflow artifact
    model_path = "tourism_project/deployment/best_tourism_model.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved successfully to {model_path}")
