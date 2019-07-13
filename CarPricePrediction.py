#Python program to predict the price of the used cars with machine learning.
#Author - Hitesh Thakur

#Importing required libraries
import pandas as pd;
from copy import deepcopy;
import matplotlib.pyplot as plt;
import numpy as np;
from sklearn.preprocessing import LabelEncoder;
from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LinearRegression;
from sklearn.metrics import r2_score,mean_squared_error;
from sklearn.preprocessing import StandardScaler;
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

#Reading data from the .csv file using pandas
df = pd.read_csv("autos.csv",encoding='cp1252');

#Dropping features that are irrelevant with price of the car [Note : No. of prictures for each sample is 0]
df.drop(['name','abtest','dateCrawled','dateCreated','lastSeen','nrOfPictures','seller','offerType','postalCode','monthOfRegistration'],axis='columns',inplace=True);

# Removing outliers
new_df=deepcopy(df[(df['price'] > 120) & (df['price'] < 120000) & (df['yearOfRegistration'] > 1970) & (df['yearOfRegistration'] < 2018) & (df['powerPS'] > 50) & (df['powerPS'] < 500)]);

#Percentage of data available after filtering and preprocessing 
print("Data remaning : ",(new_df.shape[0]/df.shape[0])*100);

#Checking for the total number of null values for each feature
print(new_df.isnull().sum());

#Replacing NaN
new_df.fillna({'vehicleType':'vt-not-declared','gearbox':'gb-not-declared','model':'m-not-declared','fuelType':'ft-not-declared','notRepairedDamage':'rd-not-declared'}, inplace=True);

#Removing duplicate samples in the dataset
new_df=new_df.drop_duplicates();

# Plotting bar plots for each feature
col_names=['vehicleType','gearbox','model','fuelType','notRepairedDamage'];
for temp in col_names:
    y=new_df[temp].groupby(new_df[temp]).count().sort_values(ascending=False);
    x=np.arange(min(len(new_df[temp].unique()),5));
    plt.bar(x,y.head()); ##Default value of head is 5
    plt.title(temp);
    plt.xticks(x,y.index, rotation=45);
    plt.show();

#Reseting index for thr preprocessed dataset
new_df.reset_index(inplace=True);

#Converting non numeric data into numeric using get_dummies function from pandas
cars=pd.DataFrame();
for x in ['vehicleType', 'gearbox','model', 'fuelType',
          'brand', 'notRepairedDamage']:
    dumpy_variable=pd.get_dummies(new_df[x],sparse=True);
    dumpy_variable.drop(dumpy_variable.columns[0],axis='columns',inplace=True);
    cars = pd.concat([cars, dumpy_variable], axis='columns');

#Features and targets
cars=pd.concat([cars,new_df['yearOfRegistration'],new_df['powerPS'],new_df['kilometer']], axis='columns');
target=new_df['price'];

#Splitting data into training set and testing set
X_train,X_test,Y_train,Y_test=train_test_split(cars,target,test_size=0.33,random_state=0);

#Normalizing data
STD=StandardScaler();
X_train=STD.fit_transform(X_train);
X_test=STD.transform(X_test);

#Training the model for our dataset
rf = RandomForestRegressor()
param_grid = { "criterion" : ["mse"]
              , "min_samples_leaf" : [3]
              , "min_samples_split" : [3]
              , "max_depth": [10]
              , "n_estimators": [60]}

gs = GridSearchCV(estimator=rf, param_grid=param_grid, cv=2, verbose=1)
gs = gs.fit(X_train, Y_train)
print(gs.best_score_)
print(gs.best_params_)
bp = gs.best_params_;
forest = RandomForestRegressor(criterion=bp['criterion'],
                              min_samples_leaf=bp['min_samples_leaf'],
                              min_samples_split=bp['min_samples_split'],
                              max_depth=bp['max_depth'],
                              n_estimators=bp['n_estimators'])
forest.fit(X_train, Y_train)
# Explained variance score: 1 is perfect prediction
print('Score: %.2f' % forest.score(X_test, Y_test));

#ADDITIONAL PREPROCESSING

##Found no correlation between price and name length; also price and abtest

#name_length=np.array([len(x) for x in new_df['name']]);
#print(name_length.shape);
#print(new_df['price'].shape);
#LE=LabelEncoder();
#new_df['abtest']=LE.fit_transform(new_df['abtest']);
#ab_merged=pd.concat([new_df['abtest'],new_df['price']],axis='columns');
#correlation_ab=ab_merged.corr();
#print(correlation_ab);
#merged_data=pd.concat([pd.DataFrame(name_length),new_df['price']],axis='columns');
#print(merged_data);
#correlation=merged_data.corr();
#print(correlation);
