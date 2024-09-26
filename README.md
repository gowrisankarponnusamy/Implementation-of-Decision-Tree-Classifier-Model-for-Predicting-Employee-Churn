# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: GOWRISANKAR P
RegisterNumber: 212222230041
```
```

import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()    #no departments and no left
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])


plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()

```

## Output:
![image](https://github.com/user-attachments/assets/66b3bde5-c50a-47ac-ada1-210cdb1535cc)

![image](https://github.com/user-attachments/assets/a64177e4-22fa-46a7-9312-973bf2eed83e)
![image](https://github.com/user-attachments/assets/547badee-603d-4976-a7d7-8e2cf7049f4d)
![image](https://github.com/user-attachments/assets/2601ec18-1fab-47b1-99ae-29da7e64fdd2)
![image](https://github.com/user-attachments/assets/71de3b61-4bc5-449f-8619-e27700f21c2c)
![image](https://github.com/user-attachments/assets/7b19dbf1-3733-427f-98aa-1a99bd59ba71)
![image](https://github.com/user-attachments/assets/da3139c5-ed4f-4b4e-a9b1-36d1d0b49101)

![image](https://github.com/user-attachments/assets/fb11c07e-e1e7-4bcb-8290-2eff98932220)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
