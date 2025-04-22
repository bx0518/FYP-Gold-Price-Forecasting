import streamlit as st
from datetime import date
import yfinance as yf
from plotly import graph_objs as go
import joblib as jl
import tensorflow as tf
import pandas as pd
import numpy as np
import locale
import plotly.express as px

model = tf.keras.models.load_model("lstm_model.h5")
scaler = jl.load("scaler.pkl")

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

START = "2005-08-31"
TODAY = pd.to_datetime(date.today().strftime("%Y-%m-%d"))

st.title("Gold Price Forecasting Dashboard")
latest, tickers = st.tabs(['Latest 30 Trading Days Data', 'Historical Data of Relevant Data'])

with latest:
    st.header("Latest 30 Trading Days Data")
    @st.cache_data
    def get_latest_30_data():
        gold = yf.download("GC=F", period="3mo", interval="1d")
        usd = yf.download("DX-Y.NYB", period="3mo", interval="1d")

        #Combine the Data
        combined_data = pd.merge(gold[["Open","High","Low","Close","Volume"]], usd["Close"],on="Date", how="inner")
        combined_data.rename(columns={"Close_x": "Close", "Close_y": "USD Index"}, inplace=True)
        latest_data = combined_data.tail(30)
        return latest_data

    def plot_latest_data():
        fig = px.line(X_train, x = X_train.index, y=X_train['Close'], title = "The Gold Price for Latest 30 Trading Days")
        st.plotly_chart(fig)

    X_train = get_latest_30_data()

    both, chart, historical = st.tabs(['Default', 'Chart', 'Historical Data'])

    with both:
        plot_latest_data()
        st.dataframe(X_train.sort_index(ascending=False))

    with chart:
        plot_latest_data()

    with historical:
        st.dataframe(X_train.sort_index(ascending=False))

    #For Prediction Part
    st.header("LSTM Model Prediction For the Next Day")

    def preprocess_input(input_data):
        input_data_scaled = scaler.fit_transform(input_data)
        return np.array(input_data_scaled).reshape((1,input_data_scaled.shape[0], input_data_scaled.shape[1]))

    def get_prediction(input_data):
        processed_data = preprocess_input(input_data)
        prediction = model.predict(processed_data)
        prediction_copy = np.repeat(prediction, 6, axis=-1)
        prediction_inverse = scaler.inverse_transform(np.reshape(prediction_copy, (len(prediction),6)))[:,0]
        return prediction_inverse

    def calculate_change(predict_value, previous_value):
        absolute_change = predict_value - previous_value
        percentage_change = (absolute_change/previous_value) * 100
        return absolute_change, percentage_change
    
    def plot_pred_data(X_train, predict_value):
        pred_df = pd.DataFrame({"Close": predict_value}, index=[X_train.index[-1] + pd.Timedelta(days=1)])
        fig = go.Figure()
        fig.add_scatter(x=X_train.index, y=X_train["Close"], mode="lines", name="Actual")
        fig.add_scatter(x=pred_df.index, y=pred_df["Close"], mode="markers", name="Prediction", marker=dict(color="red", size=10, symbol="cross"))
        fig.update_layout(title="Gold Price Prediction")
        st.plotly_chart(fig)

    if st.button('Predict for the next day'):
        prediction = get_prediction(X_train)
        last_value = X_train['Close'].iloc[-1]
        absolute_change, percentage_change = calculate_change(prediction, last_value)
        st.success("Predicted Gold Close Price: " + locale.currency(prediction[0]))
        st.info("Price Changed: " + locale.currency(absolute_change[0]))
        if absolute_change > 0:
            st.info(f'The gold price is higher by {abs(percentage_change[0]):.2f}%')
        elif absolute_change < 0:
            st.info(f'The gold price is lower by {abs(percentage_change[0]):.2f}%')
        else:
            st.info('The gold price value remains unchanged.')
        plot_pred_data(X_train, prediction)


with tickers:
    #Historical data of Gold And SP500
    st.header("Historical Data of Relevant Data")
    
    #For chosing the Dataset
    def get_datetime(date):
        return pd.to_datetime(date, format="%Y-%m-%d")

    ticker = ("GC=F", "^SPX", "CL=f", "DX-Y.NYB")
    selected_ticker = st.selectbox("Select dataset", ticker)
    start_date = st.date_input("Start Date", value=get_datetime("2005-08-31"), min_value=get_datetime("2005-08-31"), max_value=TODAY)
    end_date = st.date_input("End Date", value=TODAY, min_value=get_datetime("2005-08-31"))

    #For Loading the Data
    def load_data(ticker):
        data = yf.download(ticker, start=start_date, end=end_date)
        return data
    
    def plot_historical_data():
        fig = px.line(data, x = data.index, y=data['Close'], title = "Historical Data Chart of {0}".format(selected_ticker))
        st.plotly_chart(fig)

    data = load_data(selected_ticker)

    st.subheader("Historical Data of {0}".format(selected_ticker))
    st.dataframe(data.sort_index(ascending=False))

    plot_historical_data()