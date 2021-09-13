import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

veriler = pd.read_csv("ACME-HappinessSurvey2020.csv")
#print(veriler)

Xbir = veriler[['X1']]
#print(Xbir)

Xiki = veriler[['X2']]
#print(Xiki)

Xüç = veriler[['X3']]

Xdört = veriler[['X4']]

Xbeş = veriler[['X5']]

Xaltı = veriler[['X6']]

Y=veriler[['Y']]
#print(Y)

s= pd.concat([Xbir,Xiki],axis=1)
#print(s)

s2= pd.concat([Xüç,Xdört],axis=1)

s3= pd.concat([Xbeş,Xaltı],axis=1)

d= pd.concat([s,s2],axis=1)
#print(s)

d2= pd.concat([d,s3],axis=1)
#print(d2)


from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test= train_test_split(d2,Y,
                                                 test_size=0.33,random_state=0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train= sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

Y_train= sc.fit_transform(y_train)
Y_test=sc.fit_transform(y_test)

#d22=sc.fit_transform(d2)

from sklearn.linear_model import LinearRegression
lr= LinearRegression()
lr.fit(X_train,Y_train)

y_pred = lr.predict(X_test)
print(y_pred)

import statsmodels.api as sm
X=np.append(arr= np.ones((126,1)).astype(int),values=d2, axis=1)
Xl=d2.iloc[:,[0,1,2,3,4,5]].values
Xl=np.array(Xl,dtype=float)
model=sm.OLS(Y,Xl).fit()
print(model.summary())

Xl=d2.iloc[:,[0,1,2,4]].values
Xl=np.array(Xl,dtype=float)
model=sm.OLS(Y,Xl).fit()
print(model.summary())

Xl=d2.iloc[:,[0,1,4]].values
Xl=np.array(Xl,dtype=float)
model=sm.OLS(Y,Xl).fit()
print(model.summary())


from sklearn.metrics import mean_absolute_error
D = mean_absolute_error(Y[0:42], y_pred)
print(D)


#-----------
m= pd.concat([Xbir,Xbeş],axis=1)

x_train, x_test,y_train,y_test= train_test_split(m,Y,
                                                 test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train= sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

Y_train= sc.fit_transform(y_train)
Y_test=sc.fit_transform(y_test)

from sklearn.linear_model import LinearRegression
lr= LinearRegression()
lr.fit(X_train,Y_train)

y_pred = lr.predict(X_test)
print(y_pred)

from sklearn.metrics import mean_absolute_error
K = mean_absolute_error(Y[0:42], y_pred)
print(K)



