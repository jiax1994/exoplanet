import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
df=pd.read_csv("cumulative.csv")
df=df.drop(['rowid','kepid','kepoi_name','kepler_name','koi_disposition','koi_pdisposition','koi_period_err1','koi_period_err2','koi_time0bk_err1','koi_time0bk_err2','koi_duration_err1','koi_duration_err2','koi_depth_err1','koi_depth_err2','koi_prad_err1','koi_prad_err2','koi_teq_err1','koi_teq_err2','koi_tce_delivname','koi_steff_err1','koi_steff_err2','koi_slogg_err1','koi_slogg_err2','koi_srad_err1','koi_srad_err2'],axis=1)
df=df.dropna()
data=np.array(df)
Y=np.around(data[:,:1])
X=data[:,1:]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.5, random_state = 42)
num=X_train.shape[0]
num2=X_test.shape[0]
Y1=keras.utils.to_categorical((Y_train.ravel().astype(int))[:num], num_classes=0)

#Y2=keras.utils.to_categorical((Y_test.ravel().astype(int))[:num2], num_classes=0)
#clf=RandomForestClassifier(n_estimators=10)
#clf.fit(X_train,Y_train.ravel())

########################### RN et entrainement
model = Sequential()
model.add(Dense(units=30, input_dim=24, activation="sigmoid"))
model.add(Dropout(0.2))
model.add(Dense(units=20, activation="sigmoid"))
#model.add(Dropout(0.8))
model.add(Dense(units=10, activation="sigmoid"))
#model.add(Dropout(0.8))
model.add(Dense(units=5, activation="sigmoid"))
#model.add(Dropout(0.8))
model.add(Dense(units=2, activation="softmax"))
rmsprop=keras.optimizers.RMSprop(lr=0.01, rho=1.9, epsilon=None, decay=0.0)
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
print('########################### entrainement')
model.fit(X_train, Y1, epochs=1000, batch_size=439)


########################### prediction...
print(' ')
print(' ')
print(' ')
print(' ')
print('#############################prediction...')
tab=model.predict_classes(X_test)


########################### mesure de presicion
w=np.where(tab!=Y_test.ravel())
numw=np.array(w).shape[1]
accuracy=((num2-numw)/num2)*100
print(' ')
print(' ')
print(' ')
print(' ')
print('accuracy: ',accuracy,'%')
