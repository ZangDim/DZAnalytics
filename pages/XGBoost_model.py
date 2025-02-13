import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from navigation import make_sidebar


make_sidebar()


class XGBoostModel:
    def __init__(self, n_estimators, learning_rate, early_stopping_rounds, max_depth):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            early_stopping_rounds=early_stopping_rounds,
            max_depth=max_depth,
            n_jobs=-1
        )
        self.eval_results = None  # To store evaluation results

    def train(self, X_train, y_train, X_val, y_val):
        print('Starting XGBoost training...')
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=None)
        # self.eval_results = self.model.evals_result()

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, file_path):
        self.model.save_model(file_path)

    @staticmethod
    def load_model(file_path):
        model = xgb.XGBRegressor() 
        model.load_model(file_path)
        return model

class Tuner:
    def __init__(self, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, model_name):
        self.X_train_scaled = X_train_scaled
        self.y_train_scaled = y_train_scaled
        self.X_val_scaled = X_val_scaled
        self.y_val_scaled = y_val_scaled
        self.model_name = model_name
        

    def tune(self, n_trials):
        if self.model_name == 'XGBoostRegressor':
            best_params = self.tune_xgboost(n_trials)

        return best_params


    def tune_xgboost(self, n_trials):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 5000),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 10, 50)
            }
            model = XGBoostModel(**params)
            model.train(self.X_train_scaled,self.y_train_scaled,self.X_val_scaled,self.y_val_scaled)
            predictions = model.predict(self.X_val_scaled)
            
            return mean_absolute_percentage_error(self.y_val_scaled, predictions)
        
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=n_trials)
        print(f"Best params for XGBoost: {study.best_params}")

        return study.best_params
    

# Title and Description
st.title("XGBoost Model Trainer and Tuner")
st.write("This application allows you to upload a dataset, tune XGBoost parameters, train the model, and evaluate its performance.")

# Step 1: Data Upload
st.header("1. Preprocessed Dataset")
data = pd.DataFrame()
data = pd.read_csv('data/energy_data/Processed_dataset.csv')

if data is not None:
    st.write("### Preview of the Dataset:")
    st.write(data.head(10))

    # Allow user to select target column
    target_column = st.selectbox("Select the target column", data.columns)

    # Step 2: Parameter Selection
    st.header("2. Select Parameters to Tune")

    # XGBoost parameters for tuning
    xgboost_parameters = [
        'n_estimators', 'learning_rate', 'max_depth', 'min_child_weight', 
        'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda', 'gamma', 'early_stopping_rounds'
    ]

    selected_params = st.multiselect(
        "Select parameters to tune:", xgboost_parameters, default=['n_estimators', 'learning_rate']
    )

    st.write("### Selected Parameters:")
    st.write(selected_params)

    # Step 3: Train-Test Split
    st.header("3. Train-Test-Validation Split")
    st.write("Split your dataset into training, testing, and validation sets.")

    train_size = st.slider("Train Size (%)", 50, 80, 70, step=5) / 100
    test_size = st.slider("Test Size (%)", 10, 20, 15, step=5) / 100
    validation_size = 1-train_size-test_size

    test_size_temp = test_size / (1-train_size)

    st.write("The validation set is used for hyperparameter tuning and is created from the remaining portion of the dataset after splitting it into training and testing sets.")

    if st.button("Tune & Train Model"):
        # Prepare data
        X = data.drop(columns=[target_column])
        y = data[target_column].values.reshape(-1, 1)

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, random_state=42)
        X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        print(train_size, test_size, validation_size, test_size_temp)

        st.write("Tuning parameters...")
        tuner = Tuner(X_train, y_train, X_val, y_val, 'XGBoostRegressor')
        best_params = tuner.tune(n_trials = 100)

        st.write("### Best Parameters:")
        st.json(best_params)

        best_model = XGBoostModel(**best_params)
        best_model.train(X_train, y_train, X_val, y_val)

        # Step 4: Model Evaluation
        st.header("4. Model Evaluation")

        y_pred = best_model.predict(X_test)


        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)  # RMSE is the square root of MSE
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = ['mae', 'mse', 'rmse', 'mape', 'r2']
        metrics_values = [mae, mse, rmse, mape, r2]

        # Display the metrics
        st.write("### Metrics")
        comparison_metrics = pd.DataFrame({"Metrics": metrics, "Values": metrics_values})
        st.write(comparison_metrics.head(10))

        st.write("### Predictions vs True Values")
        comparison_pred_true = pd.DataFrame({"True Values": y_test.tolist(), "Predictions": y_pred.tolist()})
        st.write(comparison_pred_true.head(10))

        st.write("### Feature Importances")
        feature_importances = best_model.model.feature_importances_
        st.bar_chart(pd.DataFrame({"Features": X.columns, "Importance": feature_importances}).set_index("Features"))


        st.write("### Analysis")

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(y_pred, label="Predictions", linewidth=3, color='red')
        ax1.plot(y_test, label="True Values", linewidth=3, color='green')
        ax1.set_xlabel('Instances', fontsize=15)
        ax1.set_ylabel('Mean daily consumption [kWh]', fontsize=15)
        ax1.set_title('Mean daily consumption for each instance', fontsize=17)
        ax1.legend()
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.scatter(y_test, y_pred, alpha=0.6,label='Predictions vs Actual')
        ax2.plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                color='r', linestyle='--', label='y = x')
        ax2.set_xlabel('Actual Values [kWh]', fontsize=15)
        ax2.set_ylabel('Predicted Values [kWh]', fontsize=15)
        ax2.set_title('Scatter Plot: Predictions vs Actual with y = x Line', fontsize=17)
        ax2.legend()
        st.pyplot(fig2)

    else:
        st.write("Click 'Tune & Train Model' to start tuning and training.")