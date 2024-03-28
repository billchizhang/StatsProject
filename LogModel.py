import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np

# loading data from the csv file 
df = pd.read_csv('LoanPricing.csv')
# building a logreg model using sklearn 
features = ['Own','AutoPartner','InsurancePartner','Intermediary','IRR','Credit_Score','NewCar','UsedCar','Refinance','Amount','Term','Offered_Rate','Refinance','Competitor','Prime']
X = df[features]
Y = df.Accepted
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
cnf_matrix = metrics.confusion_matrix(Y_test, Y_pred)
print(cnf_matrix)
# calculate the odds ratio 
print(logreg.coef_)
odds_ratio = np.exp(logreg.coef_)
print(odds_ratio)