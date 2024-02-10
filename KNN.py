import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
data=pd.read_csv("iris.csv")
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=5
)
model=KNeighborsClassifier(10)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print('Accuraccy of KNN: \t',accuracy_score(y_pred,y_test))
pd.DataFrame({'Actutal': y_test, 'Prediction': y_pred,'Correct
classification':(y_test==y_pred)})
