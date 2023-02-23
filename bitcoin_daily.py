
import streamlit as st
from datetime import date
import yfinance as yf
import pandas as pd
import  numpy as np
import  matplotlib.pyplot as plt
from sklearn.svm import SVR
plt.style.use('fivethirtyeight')
#from pycaret.regression import *

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
#from sklearn.metrics import accuracy_score
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
st.header('bitcoin future prediction')
START='2015-01-01'
TODAY=date.today().strftime('%Y-%m-%d')
stocks=('BTC-USD','ETH-USD','AVAX-USD','LTC-USD','AAPL','GOOG','MSFT','GME')
selected_stocks=st.selectbox('selec a stock',stocks)

@st.cache_data
def load_data(ticker):
    data=yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True)
    #data=pd.DataFrame(data)
    return data
#data_load_state=st.text('load data...')
data=load_data(selected_stocks)
#data_load_state.text('loading data ...')
df_new1=pd.DataFrame()
for i in ['BTC-USD','ETH-USD','AVAX-USD','LTC-USD']:
    df_new = yf.download(i, START, TODAY)
    df_new.reset_index(inplace=True)
    df_new1[i]=df_new['Close']
st.write('btc-eth-avax-ltc close prices',df_new1)
st.write('btc-eth-avax-ltc close prices describe',df_new1.describe())
my_crypto=df_new1
plt.figure(figsize=(12.2,4.5))
for c in my_crypto.columns.values:
    plt.plot(my_crypto[c],label=c)
plt.title('crypto currency')
plt.xlabel('Days')
plt.ylabel('USD')
plt.legend()
st.pyplot()
from sklearn import preprocessing
min_max_scaler=preprocessing.MinMaxScaler(feature_range=(0,100))
scaled=min_max_scaler.fit_transform(df_new1)
st.write('scaled curenncy',scaled)
# df_scale=pd.DataFrame(scaled,df_new1.columns)
# df_scale_drop=df_scale.dropna()
# st.write('scaled dropped',df_scale_drop)
# st.markdown('---')
# my_crypto=df_scale
# plt.figure(figsize=(12.2,4.5))
# for c in my_crypto.columns.values:
#     plt.plot(my_crypto[c],label=c)
# plt.title('crypto currency')
# plt.xlabel('Days')
# plt.ylabel('USD')
# plt.legend()
# st.pyplot()



# data=data.set_index(pd.DatetimeIndex(data['Date'].values))
# st.write(data)
# st.write('data shape:',data.shape)
# plt.figure(figsize=(12,4))
# plt.plot(data['Close'])
# plt.legend()
# st.pyplot()

future_days=st.slider('select future days',1,10,value=5)
st.write('future days:',future_days)
data['Future_Price']=data[['Close']].shift(-future_days)
data=data[['Close','Future_Price']]
df=data.copy()
st.markdown('---')
st.write(df)
X=np.array(df[['Close']])
X=X[:len(df)-future_days]

y=np.array(df['Future_Price'])
y=y[:-future_days]

col1,col2=st.columns(2)
col1.write('X verileri')
col2.write('y verileri')
col1.write(X)
col2.write(y)
dsr=df['Close'].pct_change()
st.write('daily simple return',dsr)
st.write('dsr describe',dsr.describe())
plt.figure(figsize=(12,4))
plt.plot(dsr.index,dsr,lw=2)
plt.title('daily simple returns')
plt.xlabel('date')
plt.ylabel('percentage')
#plt.legend()
st.pyplot()
top=plt.subplot2grid((4,4),(0,0),rowspan=3,colspan=4)
top.plot(df.index,df['Close'],label='Close')
plt.title('Close Price')
bottom=plt.subplot2grid((4,4),(3,0),rowspan=1,colspan=4)
plt.title('DSR')
bottom.plot(dsr.index,dsr)
plt.subplots_adjust(hspace=0.75)
plt.gcf().set_size_inches(15,8)
st.pyplot()
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=227)
# model=SVR(kernel='rbf',C=1e3,gamma=0.00001)
# model.fit(X_train,y_train)
# svr_confidence=model.score(X_test,y_test)
# st.write('svr accuracy:',svr_confidence)
# # y_pred=model.predict(X_test)
# # r2=r2_score(y_test,y_pred)
# # st.write('r2 score:',r2)
# st.set_option('deprecation.showPyplotGlobalUse', False)
# svr_prediction=model.predict(X_test)
# st.write('real values:',svr_prediction)
# st.write('predicted values:',y_test)
# plt.figure(figsize=(12,4))
# fig=plt.plot(svr_prediction,label='Prediction',lw=2, alpha=0.7)
# plt.plot(y_test,label='Actual',lw=2, alpha=0.7)
# plt.xlabel('Date')
# plt.legend()
# plt.xticks(rotation=45)
# st.pyplot()
