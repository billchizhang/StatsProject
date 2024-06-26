import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import statsmodels.api as sm 
from sklearn.feature_selection import RFE

# loading data from the csv file 
df = pd.read_csv('LoanPricing.csv')
# building a logreg model using sklearn 
features = ['Own','AutoPartner','InsurancePartner','Intermediary','IRR','Credit_Score','NewCar','UsedCar','Amount',
            'Term','Offered_Rate','Competitor','Prime', 'Offered_competitor', 'Offered_prime']

X = df[features]
Y = df.Accepted
# find out the p-values 
# we select features based on recursive feature elimination of 9 features left 
rfe = RFE(LogisticRegression(solver='lbfgs', max_iter=1000), n_features_to_select=15)
rfe = rfe.fit(X, Y)
selected = rfe.support_
selected_features = [c for c, i in zip(features, selected) if i]
print("selected features based on RFE: ")
print(selected_features)
# build the logreg model with the selected features 
X_selected = df[selected_features]
X_train, X_test, Y_train, Y_test = train_test_split(X_selected, Y, test_size=0.25, random_state=42)
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train, Y_train)
# test the model accruracy and build confusion matrix 
Y_pred = logreg.predict(X_test)
cnf_matrix = metrics.confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix: ")
print(cnf_matrix)
print("Accuracy: ")
print(metrics.accuracy_score(Y_test, Y_pred))
print("Recall: ")
print(metrics.recall_score(Y_test, Y_pred))
print("Precision: ")
print(metrics.precision_score(Y_test, Y_pred))
# find out the p-values only for the completeness of the report, we are not using p-value for feature selection 
X2 = sm.add_constant(X_selected)
est = sm.OLS(Y, X2)
est2 = est.fit()
print(est2.summary())
# calculate the odds ratio to understand which factors/features are affecting the result 
print("Coefs: ")
print(logreg.coef_)
odds_ratio = np.exp(logreg.coef_)
print("Odd Ratios: ")
print(odds_ratio)

# predict using applicant pool 
applicant = pd.read_csv('ApplicantPool.csv', nrows=100)
X_app = applicant[features]
Y_app_pred = logreg.predict(X_app)
X_app['Approved'] = Y_app_pred
X_app.to_csv('PredictedApporval.csv')