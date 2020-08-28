from sklearn.model_selection import train_test_split
import Config 
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import sklearn



class Pipeline:

    def __init__(self,target,features,data,split_pct):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        self.target=target
        self.features=features
        self.data=data
        self.split_pct=split_pct

        self.model = Lasso(alpha=0.005, random_state=0)


    def split_data(self,data):
        x=data[self.features]
        y=data[self.target]
        self.split_pct=Config.split_pct
        self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(x,y,test_size=self.split_pct,random_state=0)


    def Impute_Categories(self,data,features):
        self.data[features]=self.data[features].fillna('Missing')

    def Impute_numerics(self,x_train,x_test,features):
        for var in features:
            value=x_train[var].mode()[0]
    
            self.x_train[var+'_na']= np.where(self.x_train[var].isnull(),1,0)
    
            self.x_test[var+'_na']= np.where(self.x_test[var].isnull(),1,0)
    
            self.x_train[var]=self.x_train[var].fillna(value)
            self.x_test[var]=self.x_test[var].fillna(value)

    def Year_Value_To_NumOfYears(self,x_train,x_test,features):
        for var in features:
            self.x_train[var]=self.x_train['YrSold']-self.x_train[var]
            self.x_test[var]=self.x_test['YrSold']-self.x_test[var]
    def logtransform(self,data,features):
        for var in features:
            self.data[var]=np.log(self.data[var])

    def Categorical_Encoding(self,data,features):
        data_catEncoded=data[features].apply(LabelEncoder().fit_transform)
        data.drop(features,axis=1,inplace=True)
        return pd.concat([data,data_catEncoded],axis=1)
        
    def featurescaling(self,x_train,x_test):
        scaler=MinMaxScaler()
        scaler.fit(x_train)
        self.x_train=scaler.transform(self.x_train)
        self.x_test=scaler.transform(self.x_test)


    def fit(self,data):
        
        self.logtransform(self.data,Config.LogTransormFeatures)

        self.Impute_Categories(data,Config.categories_to_Impute)

        self.split_data(data)

        self.Impute_numerics(self.x_train,self.x_test,Config.Numerics_to_impute)

        self.Year_Value_To_NumOfYears(x_train=self.x_train,x_test=self.x_test,features=Config.Years_toBe_Transformed)

        self.x_train=self.Categorical_Encoding(data=self.x_train,features=Config.categorical_variables)
        self.x_test=self.Categorical_Encoding(data=self.x_test,features=Config.categorical_variables)

        self.featurescaling(self.x_train,self.x_test)

        self.model.fit(self.x_train,np.log(self.y_train))

        return self

    def evaluate(self):
        pred=self.model.predict(self.x_train)
        pred=np.exp(pred)

        print('train r2 : {}  '.format(r2_score(self.y_train,pred)))

        pred=self.model.predict(self.x_test)
        pred=np.exp(pred)
        print('test r2 : {} '.format(r2_score(self.y_test,pred)))








    

