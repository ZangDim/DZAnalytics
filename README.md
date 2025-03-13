# DZAnalytics

**DZAnalytics** is a web application that analyzes the **Sakila Database**, extracts datasets, and executes pre-trained machine learning (ML) models. It provides users with a platform for advanced data analysis, enabling valuable insights through an intuitive and interactive interface.

---

## Deployment

**DZAnalytics** is hosted on the **Streamlit Community Cloud**, allowing users to access and interact with the application directly through their web browsers. There’s no need for any local setup or installations—just visit the link and start using the app.

🔗 **Access the app here:** [DZAnalytics](https://dzanalytics.streamlit.app/)

---

## Energy Consumption Model

The **Energy Consumption Model** page is designed to help users train and fine-tune an XGBoost model by selecting key parameters. Users can adjust the following:

- **Tuning Parameters**: Choose which hyperparameters to tune for better model performance.
- **Model Parameters**: Set essential attributes of the XGBoost model.
- **Train-Test Split**: Define the percentage of data for training and testing.
- **Target Column Selection**: Specify the target variable for predictions.

This section helps you to quickly train, tune, and evaluate a model on your dataset with a straightforward interface.

📝 **Note**: The tuner’s parameters have been simplified for faster execution, which may result in slightly less accurate results compared to a fully optimized model.

---

## Steps to Use the Energy Consumption Model

1. **Select the Target Column**: Choose the `label` column (the dependent variable for prediction).
2. **Choose Parameters to Tune**: Select from `n_estimators`, `learning_rate`, `max_depth`, and `early_stopping_rounds`.
3. **Set Train-Test Split**: Use a 70% training and 15% testing split for optimal performance (default).
4. **Tune and Train the Model**: Press the **"Tune & Train Model"** button to initiate the training process and view the results.

---

## Performance Considerations

- **App Startup**: If the app has been inactive for some time, it may take a few seconds to restart. Please be patient as it loads.
  
Thank you for using **DZAnalytics**!! 🚀
