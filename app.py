import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load Data
st.title('Stock Price Prediction')
st.write("This app is used to predict stock prices.")

# Upload CSV Data
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Show a preview of the data
    st.write("Preview of the dataset:")
    st.write(data.head())

    # Feature Selection
    st.write("### Select Features and Target")
    features = st.multiselect("Select feature columns", options=data.columns.tolist())
    target = st.selectbox("Select target column", options=[col for col in data.columns if col not in features])

    if len(features) > 0 and target:
        X = data[features]
        y = data[target]

        # Train-Test Split
        test_size = st.slider('Test size (percentage)', min_value=10, max_value=50, value=20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

        # Model Training
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Prediction
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Model Evaluation
        st.write("### Model Evaluation")
        st.write(f"Train R^2 Score: {r2_score(y_train, y_train_pred):.2f}")
        st.write(f"Test R^2 Score: {r2_score(y_test, y_test_pred):.2f}")
        st.write(f"Test Mean Squared Error: {mean_squared_error(y_test, y_test_pred):.2f}")

        # Plotting Predictions
        st.write("### Predictions vs Actual")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_test_pred, edgecolors=(0, 0, 0))
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)

        # User Prediction
        st.write("### Make Predictions")
        user_input = []
        for feature in features:
            value = st.number_input(f"Enter {feature}", value=float(data[feature].mean()))
            user_input.append(value)

        if st.button("Predict"):
            prediction = model.predict([user_input])[0]
            st.write(f"Predicted {target}: {prediction}")
