import pandas as pd
import matplotlib.pyplot as mplt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
df=pd.read_csv('C:/Users/pushp/Downloads/train.csv (1).zip')
le=LabelEncoder()
ss=StandardScaler()
df['POSTED_BY']=le.fit_transform(df['POSTED_BY'])
df['BHK_OR_RK']=le.fit_transform(df['BHK_OR_RK'])
df['ADDRESS']=le.fit_transform(df['ADDRESS'])
df['LONGITUDE']=ss.fit_transform(df[['LONGITUDE']])
df['LATITUDE']=ss.fit_transform(df[['LATITUDE']])
df['SQUARE_FT']=ss.fit_transform(df[['SQUARE_FT']])
df['TARGET(PRICE_IN_LACS)']=ss.fit_transform(df[['TARGET(PRICE_IN_LACS)']])
x=df.iloc[:,0:11]
y=df.iloc[:,11:12]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
lr=LinearRegression()
lr.fit(x_train,y_train)
a=lr.score(x_test,y_test)*100
print(a,"%")
dtr=DecisionTreeRegressor(max_features=8,max_leaf_nodes=6)
dtr.fit(x_train,y_train)
b=dtr.score(x_test,y_test)*100
print(b,"%")
rfr=RandomForestRegressor(n_estimators=8,max_features=8,max_leaf_nodes=6)
rfr.fit(x_train,y_train)
c=rfr.score(x_test,y_test)*100
print(c,"%")
gbr=GradientBoostingRegressor(n_estimators=15,loss='squared_error')
gbr.fit(x_train,y_train)
d=gbr.score(x_test,y_test)
print(d*100,"%")