import streamlit as st
from datetime import date
import yfinance as yf
import pandas as pd

from fbprophet import Prophet#_inference.forecaster import Prophet
from plotly import graph_objs as go
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
st.title('stock prediction app')
st.header('stock prediction')
START='2015-01-01'
TODAY=date.today().strftime('%Y-%m-%d')
stocks=('AAPL','GOOG','MSFT','GME')
selected_stocks=st.selectbox('selec a stock',stocks)
n_years=st.slider('yers of prediction',1,4)
period=n_years*365

@st.cache_data
def load_data(ticker):
    data=yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True)
    #data=pd.DataFrame(data)
    return data
data_load_state=st.text('load data...')
data=load_data(selected_stocks)
data_load_state.text('loading data ...')
st.dataframe(data)
st.write(data.shape)
def plot_raw_data():
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text='time series data',xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
plot_raw_data()

df_train=data[['Date','Close']]
df_train=df_train.rename(columns={"Date":'x','Close':'y'})
m=Prophet()
m.fit(df_train)
future=m.make_future_dataframe(periods=period)
forecast=m.predict(future)
st.subheader('Forecast Data')
st.write(forecast.tail())

