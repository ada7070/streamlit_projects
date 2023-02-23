import streamlit as st
import pandas as pd
import  numpy as np
import  matplotlib.pyplot as plt
import seaborn as sns
#import tensorflow as tf
#plt.style.use('fivethirtyeight')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
import io  # data.info() nun gorulmesi icin gerekli

sns.set_style('darkgrid')


st.header('heart desease prediction')
# @st.cache_data
# def load_data():
#     data=pd.read_csv('diabetes_data_upload.csv')
#     return data
# data=load_data()
# st.write(data)
# st.write(data.shape)
# st.write('missing values:',data.isna().sum().sum())
# #pandas data.info() komutunun calismasi icin asagidaki satirlar gerekli
# buffer = io.StringIO()
# data.info(buf=buffer)
# s = buffer.getvalue()
# st.text(s)
# st.write({column:len(data[column].unique()) for column in data.columns})# calculates each coulmns unique values
# # # pandas data.info() komutunun calismasi icin yukaridaki satirlar gerekli
# def preprocess_inputs(df):
#     df=df.copy()
#     df['Gender']=df['Gender'].replace({'Female':0,'Male':1})
#     for column in df.columns.drop(['Age','Gender','class']):
#         df[column]=df[column].replace({'No':0,'Yes':1})
#     y=df['class']
#     X=df.drop('class',axis=1)
#     X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.7,shuffle=1,random_state=1)
#     scaler=StandardScaler()
#     scaler.fit(X_train)
#     X_train=pd.DataFrame(scaler.transform(X_train),index=X_train.index,columns=X_train.columns)
#     X_test=pd.DataFrame(scaler.transform(X_test),index=X_test.index,columns=X_test.columns)
#
#     return X_train,X_test,y_train,y_test
# X_train,X_test,y_train,y_test=preprocess_inputs(data)
# st.write('X_train:',X_train)
# st.write('X_train shape:',X_train.shape)
# models={
#     'Logistic Regression':LogisticRegression(),
#     'K Nearest Neighbors':KNeighborsClassifier(),
#     'Support Vector Machine Linear':LinearSVC(),
#     'Support Vector Machine':SVC(),
#     'Decision Tree':DecisionTreeClassifier(),
#     'Neural Network':MLPClassifier(),
#     'Random Forest':RandomForestClassifier(),
#     'Gradient Boosting':GradientBoostingClassifier()
# }
# for name,model in models.items():
#     model.fit(X_train,y_train)
#     #st.write(name+' trained')
# st.markdown('---')
# for name,model in models.items():
#     st.write(f'{name}:',model.score(X_test,y_test))
# st.markdown('---')

# data=data.drop(['id'],axis=1)
# st.write(data)
# st.write(data.shape)
#
# y=data['diagnosis'].copy()
# X=data.drop('diagnosis',axis=1).copy()
#
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.7,random_state=123)
# scaler=StandardScaler()
# scaler.fit(X_train)
# X_train=pd.DataFrame(scaler.transform(X_train),columns=X_train.columns)
# X_test=pd.DataFrame(scaler.transform(X_test),columns=X_train.columns)
# st.write(X_train)
#
# original_model=LogisticRegression()
# original_model.fit(X_train,y_train)
# st.write('original model score:',original_model.score(X_test,y_test))
# st.markdown('---')
# st.write(data.corr())
# n_components=st.slider('n_components',1,(X_train.shape)[1],2)
# pca=PCA(n_components=n_components)
# pca.fit(X_train)
# pc_train=pd.DataFrame(pca.transform(X_train),columns=["PC"+str(i+1) for i in range(n_components)])
# pc_test=pd.DataFrame(pca.transform(X_test),columns=["PC"+str(i+1) for i in range(n_components)])
# st.write(pc_train)
# st.set_option('deprecation.showPyplotGlobalUse', False)
# plt.figure(figsize=(16,10))
# sns.barplot(x=pca.explained_variance_ratio_,y=["PC"+str(i+1) for i in range(n_components)])
# st.pyplot()
#
# pca_model=LogisticRegression()
# pca_model.fit(pc_train,y_train)
# st.write('pca model score:',pca_model.score(pc_test,y_test))
# st.markdown('---')
# # fig, ax = plt.subplots()
# # ax.scatter([1, 2, 3], [1, 2, 3])
# # ... other plotting actions ...
# # st.pyplot(fig)
# # CORR MATRIX----
# # corr_matrix=train.corr()
# # feature = 'imp_op_var39_comer_ult1'
# # (corr_matrix[feature].iloc[:corr_matrix.columns.get_loc(feature)] > 0.8).any()
# #
# # corr_features=[feature for feature in corr_matrix.columns if(corr_matrix[feature].iloc[:corr_matrix.columns.get_loc(feature)] > 0.8).any() ]
# dfCorr = X_train.corr()
# filteredDf = dfCorr[((dfCorr >= .5) | (dfCorr <= -.5)) & (dfCorr !=1.000)]
# plt.figure(figsize=(30,10))
# sns.heatmap(filteredDf, annot=True, cmap="Reds")
# st.pyplot()
# st.write('filtered correlation:',filteredDf)
# # CORR MATRIX END----
#
#
# # #tf.keras.utils.plot_model(model)
# # model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# # batch_size=32
# # epochs=50
# # history=model.fit(X_train,y_train,
# #                   validation_split=0.2,
# #                   batch_size=batch_size,
# #                   epochs=epochs,
# #                   callbacks=[tf.keras.callbacks.EarlyStopping(monitor='vall_cross',
# #                                                               patience=3,
# #                                                               restore_best_weights=True)])
# #
# # model_acc=model.evaluate(X_test,y_test,verbose=0)[1]
# # st.write('model accuracy:',model_acc)
# # y_true=np.array(y_test)
# # y_pred=model.predict(X_test)
# # y_pred=np.array(list(map(lambda x:np.argmax(x),y_pred)))
# # st.write('real:',y_true)
# # st.write('predicted:',y_pred)
# #
# # cm=confusion_matrix(y_true,y_pred)
# # clr=classification_report(y_true,y_pred)
# # st.write('confusion matrix:',cm)
# # st.write('classification report:',y_true,y_pred,target_names=label_mapping.values())
# # #plt.figure(figsize=(8,8))
# # sns.heatmap(cm,annot=True,fmt='g',vmin=0,cbar=False,cmap='Blues')
# # plt.xlabel('predicted')
# # plt.ylabel('actual')
# # plt.xticks(np.arange(7)+0.5,label_mapping.values())
# # plt.yticks(np.arange(7)+0.5,label_mapping.values())
# # st.pyplot()
# # st.write(clr)

st.write('---End---')