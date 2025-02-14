# import xgboost as xgb
# import optuna
# import matplotlib.pyplot as plt
# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# import plotly.express as px
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
# from navigation import make_sidebar


# make_sidebar()


# class XGBoostModel:
#     def __init__(self, n_estimators, learning_rate, early_stopping_rounds, max_depth):
#         self.model = xgb.XGBRegressor(
#             n_estimators=n_estimators,
#             learning_rate=learning_rate,
#             early_stopping_rounds=early_stopping_rounds,
#             max_depth=max_depth,
#             n_jobs=-1
#         )
#         self.eval_results = None  # To store evaluation results

#     def train(self, X_train, y_train, X_val, y_val):
#         print('Starting XGBoost training...')
#         self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=None)
#         # self.eval_results = self.model.evals_result()

#     def predict(self, X):
#         return self.model.predict(X)

#     def save_model(self, file_path):
#         self.model.save_model(file_path)

#     @staticmethod
#     def load_model(file_path):
#         model = xgb.XGBRegressor() 
#         model.load_model(file_path)
#         return model

# class Tuner:
#     def __init__(self, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, model_name):
#         self.X_train_scaled = X_train_scaled
#         self.y_train_scaled = y_train_scaled
#         self.X_val_scaled = X_val_scaled
#         self.y_val_scaled = y_val_scaled
#         self.model_name = model_name
        

#     def tune(self, n_trials):
#         if self.model_name == 'XGBoostRegressor':
#             best_params = self.tune_xgboost(n_trials)

#         return best_params


#     def tune_xgboost(self, n_trials):
#         def objective(trial):
#             params = {
#                 'n_estimators': trial.suggest_int('n_estimators', 100, 5000),
#                 'learning_rate': trial.suggest_float('learning_rate', 1e-5, 0.3),
#                 'max_depth': trial.suggest_int('max_depth', 3, 15),
#                 'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 10, 50)
#             }
#             model = XGBoostModel(**params)
#             model.train(self.X_train_scaled,self.y_train_scaled,self.X_val_scaled,self.y_val_scaled)
#             predictions = model.predict(self.X_val_scaled)
            
#             return mean_absolute_percentage_error(self.y_val_scaled, predictions)
        
#         study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
#         study.optimize(objective, n_trials=n_trials)
#         print(f"Best params for XGBoost: {study.best_params}")

#         return study.best_params
    

# # Title and Description
# st.title("XGBoost Model Trainer and Tuner")
# st.write("This application allows you to upload a dataset, tune XGBoost parameters, train the model, and evaluate its performance.")

# # Step 1: Data Upload
# st.header("1. Preprocessed Dataset")
# data = pd.DataFrame()
# data = pd.read_csv('data/energy_data/Processed_dataset.csv')

# if data is not None:
#     st.write("### Preview of the Dataset:")
#     st.write(data.head(10))

#     # Allow user to select target column
#     target_column = st.selectbox("Select the target column", data.columns)

#     # Step 2: Parameter Selection
#     st.header("2. Select Parameters to Tune")

#     # XGBoost parameters for tuning
#     xgboost_parameters = [
#         'n_estimators', 'learning_rate', 'max_depth', 'min_child_weight', 
#         'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda', 'gamma', 'early_stopping_rounds'
#     ]

#     selected_params = st.multiselect(
#         "Select parameters to tune:", xgboost_parameters, default=['n_estimators', 'learning_rate']
#     )

#     st.write("### Selected Parameters:")
#     st.write(selected_params)

#     # Step 3: Train-Test Split
#     st.header("3. Train-Test-Validation Split")
#     st.write("Split your dataset into training, testing, and validation sets.")

#     train_size = st.slider("Train Size (%)", 50, 80, 70, step=5) / 100
#     test_size = st.slider("Test Size (%)", 10, 20, 15, step=5) / 100
#     validation_size = 1-train_size-test_size

#     test_size_temp = test_size / (1-train_size)

#     st.write("The validation set is used for hyperparameter tuning and is created from the remaining portion of the dataset after splitting it into training and testing sets.")

#     if st.button("Tune & Train Model"):
#         # Prepare data
#         X = data.drop(columns=[target_column])
#         y = data[target_column].values.reshape(-1, 1)

#         X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, random_state=42)
#         X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#         print(train_size, test_size, validation_size, test_size_temp)

#         st.write("Tuning parameters...")
#         tuner = Tuner(X_train, y_train, X_val, y_val, 'XGBoostRegressor')
#         best_params = tuner.tune(n_trials = 100)

#         st.write("### Best Parameters:")
#         st.json(best_params)

#         best_model = XGBoostModel(**best_params)
#         best_model.train(X_train, y_train, X_val, y_val)

#         # Step 4: Model Evaluation
#         st.header("4. Model Evaluation")

#         y_pred = best_model.predict(X_test)


#         mae = mean_absolute_error(y_test, y_pred)
#         mse = mean_squared_error(y_test, y_pred)
#         rmse = np.sqrt(mse)  # RMSE is the square root of MSE
#         mape = mean_absolute_percentage_error(y_test, y_pred)
#         r2 = r2_score(y_test, y_pred)

#         metrics = ['mae', 'mse', 'rmse', 'mape', 'r2']
#         metrics_values = [mae, mse, rmse, mape, r2]

#         # Display the metrics
#         st.write("### Metrics")
#         comparison_metrics = pd.DataFrame({"Metrics": metrics, "Values": metrics_values})
#         st.write(comparison_metrics.head(10))

#         st.write("### Predictions vs True Values")
#         comparison_pred_true = pd.DataFrame({"True Values": y_test.tolist(), "Predictions": y_pred.tolist()})
#         st.write(comparison_pred_true.head(10))

#         st.write("### Feature Importances")
#         feature_importances = best_model.model.feature_importances_
#         st.bar_chart(pd.DataFrame({"Features": X.columns, "Importance": feature_importances}).set_index("Features"))


#         st.write("### Analysis")

#         # Ensure valid, non-NaN, finite data
#         y_test = y_test[np.isfinite(y_test)]
#         y_pred = y_pred[np.isfinite(y_pred)]

#         # First Plot: Line plot for Predictions vs. True Values
#         fig1 = go.Figure()
#         fig1.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred, 
#                                 mode='lines', name='Predictions', 
#                                 line=dict(color='red', width=3)))
#         fig1.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test, 
#                                 mode='lines', name='True Values', 
#                                 line=dict(color='green', width=3)))  # Ensure green line is visible

#         fig1.update_layout(
#             title='Mean daily consumption for each instance',
#             xaxis_title='Instances',
#             yaxis_title='Mean daily consumption [kWh]',
#             legend_title='Legend',
#             font=dict(size=15)
#         )

#         # Display the figure in Streamlit
#         st.plotly_chart(fig1)

#         # Second Plot: Scatter plot with y = x line
#         fig2 = go.Figure()

#         # Scatter plot for Predictions vs Actual with increased marker size and full opacity
#         fig2.add_trace(go.Scatter(x=y_test, y=y_pred, 
#                                 mode='markers', 
#                                 name='Predictions vs Actual', 
#                                 opacity=1, 
#                                 marker=dict(size=8)))  # Increased marker size for visibility

#         # Line for y = x (ensures the line spans the full range of the data)
#         fig2.add_trace(go.Scatter(x=[min(y_test), max(y_test)], 
#                                 y=[min(y_test), max(y_test)], 
#                                 mode='lines', 
#                                 name='y = x', 
#                                 line=dict(color='red', dash='dash')))

#         # Ensure axis ranges are appropriate for better visibility
#         fig2.update_xaxes(range=[0, max(y_test.max(), y_pred.max())])
#         fig2.update_yaxes(range=[0, max(y_test.max(), y_pred.max())])

#         fig2.update_layout(
#             title='Scatter Plot: Predictions vs Actual with y = x Line',
#             xaxis_title='Actual Values [kWh]',
#             yaxis_title='Predicted Values [kWh]',
#             legend_title='Legend',
#             font=dict(size=15)
#         )
#         # Display the figure in Streamlit
#         st.plotly_chart(fig2)
#     else:
#         st.write("Click 'Tune & Train Model' to start tuning and training.")

import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from navigation import make_sidebar

make_sidebar()

# Caching data to avoid reloading
@st.cache_data
def load_data():
    return pd.read_csv('data/energy_data/Processed_dataset.csv')

# Caching model to avoid rebuilding
@st.cache_resource
def create_xgboost_model(params):
    return xgb.XGBRegressor(**params)

class XGBoostModel:
    def __init__(self, n_estimators, learning_rate, early_stopping_rounds, max_depth):
        self.model = create_xgboost_model({
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'early_stopping_rounds': early_stopping_rounds,
            'max_depth': max_depth,
            'n_jobs': -1,
            'tree_method': 'hist'  # Fast, efficient CPU training
        })

    def train(self, X_train, y_train, X_val, y_val):
        print('Starting XGBoost training...')
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=None)

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, file_path):
        self.model.save_model(file_path)

    @staticmethod
    def load_model(file_path):
        model = create_xgboost_model({})
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
                'n_estimators': trial.suggest_int('n_estimators', 50, 100),  # Reduced range
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),  # Narrower range
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 10, 25)
            }
            model = XGBoostModel(**params)
            model.train(self.X_train_scaled, self.y_train_scaled, self.X_val_scaled, self.y_val_scaled)
            predictions = model.predict(self.X_val_scaled)
            return mean_absolute_percentage_error(self.y_val_scaled, predictions)

        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)  # Parallel processing
        print(f"Best params for XGBoost: {study.best_params}")
        return study.best_params

# Title and Description
st.title("XGBoost Model Trainer and Tuner")
st.write("This application allows you to upload a dataset, tune XGBoost parameters, train the model, and evaluate its performance.")

# Step 1: Data Upload
st.header("1. Preprocessed Dataset")
data = load_data()
st.write("### Preview of the Dataset:")
st.write(data.head(10))

# Allow user to select target column
target_column = st.selectbox("Select the target column", data.columns)

# Step 2: Parameter Selection
st.header("2. Select Parameters to Tune")
xgboost_parameters = [
    'n_estimators', 'learning_rate', 'max_depth', 'min_child_weight', 
        'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda', 'gamma', 'early_stopping_rounds'
]

selected_params = st.multiselect("Select parameters to tune:", xgboost_parameters, default=['n_estimators', 'learning_rate'])
st.write("### Selected Parameters:")
st.write(selected_params)

# Step 3: Train-Test Split
st.header("3. Train-Test-Validation Split")
train_size = st.slider("Train Size (%)", 50, 80, 70, step=5) / 100
test_size = st.slider("Test Size (%)", 10, 20, 15, step=5) / 100
validation_size = 1 - train_size - test_size

# Start tuning and training
if st.button("Tune & Train Model"):
    # Data Preparation
    X = data.drop(columns=[target_column])
    y = data[target_column].values.reshape(-1, 1)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_size, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    st.write("Tuning parameters...")
    tuner = Tuner(X_train, y_train, X_val, y_val, 'XGBoostRegressor')
    best_params = tuner.tune(n_trials=50)  # Reduced trials for faster tuning
    st.write("### Best Parameters:")
    st.json(best_params)

    best_model = XGBoostModel(**best_params)
    best_model.train(X_train, y_train, X_val, y_val)

    # Step 4: Model Evaluation
    st.header("4. Model Evaluation")
    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = ['mae', 'mse', 'rmse', 'mape', 'r2']
    metrics_values = [mae, mse, rmse, mape, r2]
    st.write("#### Metrics")
    comparison_metrics = pd.DataFrame({"Metrics": metrics, "Values": metrics_values})
    st.write(comparison_metrics)

    # Visualization
    st.write("#### Predictions vs True Values")
    comparison_pred_true = pd.DataFrame({"True Values": y_test.flatten(), "Predictions": y_pred.flatten()})
    st.write(comparison_pred_true.head(10))

    # Line plot
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred, mode='lines', name='Predictions', line=dict(color='red', width=3)))
    fig1.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test.flatten(), mode='lines', name='True Values', line=dict(color='green', width=3)))
    fig1.update_layout(title='Mean daily consumption for each instance', xaxis_title='Instances', yaxis_title='Mean daily consumption [kWh]')
    st.plotly_chart(fig1)

        # Scatter plot
    fig2 = go.Figure()

    # Ensure valid, flattened data
    y_test_flat = y_test.flatten()
    y_pred_flat = y_pred.flatten()

    # Scatter plot for Predictions vs Actual
    fig2.add_trace(go.Scatter(
        x=y_test_flat, 
        y=y_pred_flat, 
        mode='markers', 
        name='Predictions vs Actual', 
        opacity=1, 
        marker=dict(size=8)
    ))

    # Ensure y = x line spans the full range of the data
    min_val = min(min(y_test_flat), min(y_pred_flat))
    max_val = max(max(y_test_flat), max(y_pred_flat))

    # Add the y = x line
    fig2.add_trace(go.Scatter(
        x=[min_val, max_val], 
        y=[min_val, max_val], 
        mode='lines', 
        name='y = x', 
        line=dict(color='red', dash='dash')
    ))

    # Layout adjustments
    fig2.update_layout(
        title='Predictions vs Actual with y = x Line',
        xaxis_title='Actual Values [kWh]',
        yaxis_title='Predicted Values [kWh]',
        showlegend=True
    )

    # Display in Streamlit
    st.plotly_chart(fig2)

else:
    st.write("Click 'Tune & Train Model' to start tuning and training.")
