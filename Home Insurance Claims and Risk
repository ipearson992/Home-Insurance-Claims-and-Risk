import pandas as pd
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Path to your CSV in the Documents folder
csv_path = '/storage/emulated/0/Documents/home_insurance_dataset.csv'

df = pd.read_csv(csv_path)

conn = sqlite3.connect('home_insurance.db')
table_name = 'home_insurance'
df.to_sql(table_name, conn, if_exists = 'replace', index = False)

# Number of policies for each home type
home_type = df['Home_Type']
num_home_type = home_type.value_counts()
print("Number of policices per home type:", "\n", pd.DataFrame(num_home_type))
print("\n")

#Total claims in the last 5 years
claims_list = df['Number_of_Claims_Past_5_Years']
total_claims = claims_list.sum()
print("Total number of claims:", "\n", total_claims)
print("\n")

#policies per state
state = df['State']
num_per_state = state.value_counts()
print("Number of policies per state:", "\n", pd.DataFrame(num_per_state))
print("\n")

# Premium Difference From The Mean
ave_premium = round(df['Annual_Premium'].mean(), 2)
print("The average premium was:", ave_premium)

df_prem_diff = pd.DataFrame(columns = ['Difference from mean'])
for value in df['Annual_Premium']:
    mean_diff = ave_premium - value
    df_prem_diff=pd.concat([df_prem_diff, pd.DataFrame({'Difference from mean' :[mean_diff]})], ignore_index=True)

abs_diff = round(df_prem_diff['Difference from mean'].abs(), 2)
print("The maximum premium difference from the mean was:", abs_diff.max(),"\n", "The minimum premium difference from the mean was:", abs_diff.min())
print("\n")


#High Risk Policies
df_high_risk =  pd.read_sql_query("SELECT Policy_ID, Number_of_Claims_Past_5_Years FROM home_insurance WHERE Number_of_Claims_Past_5_Years >2", conn )
print("High risk poilicies with more than 2 claims in the past 5 years:","\n", df_high_risk)
print("\n")


#Low credit Policies
df_low_credit = pd.read_sql_query("SELECT Policy_ID, Credit_Score FROM home_insurance WHERE Credit_Score <670", conn)
print("Policies of clients with fair or poor credit:", "\n", df_low_credit)
print("\n")

#High Value Homes
df_high_value_homes = pd.read_sql_query("SELECT Policy_ID, Home_Value FROM home_insurance WHERE Home_Value >= 500000", conn)
print("Policies on homes with values greater than or equal to $500,000:", "\n", df_high_value_homes)
print("\n")


#df_basic isolates the "basic" policies and therespective deductibles, premiums, and house value
df_basic = pd.read_sql_query("SELECT Coverage_Level, Deductible, Annual_Premium, Home_Value FROM home_insurance WHERE Coverage_Level LIKE 'basic' ", conn)
basic_average_deductible = round(df_basic['Deductible'].mean(), 2)
basic_average_premium = round(df_basic['Annual_Premium'].mean(), 2)
basic_average_home_value = round(df_basic['Home_Value'].mean(), 2)
print("Basic Policies:", "\n", "Average Deductible:", basic_average_deductible, "\n" , "Average Premium:", basic_average_premium, "\n", "Average Home Value:", basic_average_home_value)
print("\n")


#df_standard isolates the "Standard" policies and therespective deductibles, premiums, and house value
df_standard = pd.read_sql_query("SELECT Coverage_Level, Deductible, Annual_Premium, Home_Value FROM home_insurance WHERE Coverage_Level LIKE 'Standard' ", conn)
standard_average_deductible = round(df_standard['Deductible'].mean(), 2)
standard_average_premium = round(df_standard['Annual_Premium'].mean(), 2)
standard_average_home_value = round(df_standard['Home_Value'].mean(), 2)
print("Standard Policies:", "\n", "Average Deductible:", standard_average_deductible, "\n" , "Average Premium:", standard_average_premium, "\n", "Average Home Value:", standard_average_home_value)
print("\n")

#df_premium isolates the "premium" policies and therespective deductibles, premiums, and house value
df_premium = pd.read_sql_query("SELECT Coverage_Level, Deductible, Annual_Premium, Home_Value FROM home_insurance WHERE Coverage_Level LIKE 'premium' ", conn)
premium_average_deductible = round(df_premium['Deductible'].mean(), 2)
premium_average_premium = round(df_premium['Annual_Premium'].mean(), 2)
premium_average_home_value = round(df_premium['Home_Value'].mean(), 2)
print("Premium Policies:", "\n", "Average Deductible:", premium_average_deductible, "\n" , "Average Premium:", premium_average_premium, "\n", "Average Home Value:", premium_average_home_value)
print("\n")


#Form a regression model based on number of claims   
df['Number_of_Claims_Past_5_Years'] = df['Number_of_Claims_Past_5_Years'].astype(int)
claim_list = []
for claim in df['Number_of_Claims_Past_5_Years']:
    if claim > 2:
        claim_list.append(1)
    else:
        claim_list.append(0)
df['High_Risk'] = claim_list

features = ['Annual_Premium', 'Deductible', 'Home_Value', 'Credit_Score']
X = df[features]
y = df['High_Risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))

coefficients = pd.DataFrame({'Feature': features, 'Coefficients': model.coef_[0]})
print("The correlation of the below features to the risk of the policy:", pd.DataFrame(coefficients), "\n")

#Forest Classification to see factors' contribuations to risk'
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_forest = rf_model.predict(X_test)

#print(confusion_matrix(y_test, y_pred_forest))
#print(classification_report(y_test, y_pred_forest))

importance_df = pd.DataFrame({'Feature': features, 'Importance': rf_model.feature_importances_}).sort_values(by='Importance', ascending=False)

print("The contribution of the below features to the risk of the policy:", pd.DataFrame(importance_df), "\n")


#Probability of risk
risk_probs = rf_model.predict_proba(X_test)
high_risk_probs = risk_probs[:, 1] 
results = X_test.copy()
results['Actual'] = y_test.values
results['Predicted'] = rf_model.predict(X_test)
results['Predicted_Risk'] = high_risk_probs

high_risk_ranked = results.sort_values(by='Predicted_Risk', ascending=False)

print(high_risk_ranked)
print("\n")
#Finding over/under priced policy. In this case, overpriced is defined predicted risk <= .13 and premium >= 2000. Underpriced will be defined as predicted risk> .14 and premium <= 2000.
df_overpriced = high_risk_ranked[(high_risk_ranked['Annual_Premium'] >=2000) & (high_risk_ranked['Predicted_Risk'] <= .13)].copy()
print("Overpriced policies:", df_overpriced, "\n")

df_underpriced = high_risk_ranked[(high_risk_ranked['Annual_Premium'] <= 2000) & (high_risk_ranked['Predicted_Risk'] >= .13)].copy()
print("Underpriced policies:", df_underpriced, "\n")



#find the average number of claims given that the house has a). no pool and no trampoline, b). a pool but no trampoline, c). no pool but a trampoline, d). both a pool and a trampoline where pool = "p" and trampoline ="t"
#a).
df_no_pt = pd.read_sql_query("SELECT Number_of_Claims_Past_5_Years FROM home_insurance WHERE Has_Pool ='No' AND Has_Trampoline ='No' ", conn)
num_rows_no_pt = len(df_no_pt)
claims_no_pt = df_no_pt.sum().values[0] 
ave_claims_no_pt = round(claims_no_pt/num_rows_no_pt, 2)
print("The average number of claims, given that the house has no pool or trampoline, is", ave_claims_no_pt,"\n")

#b).
df_p_no_t = pd.read_sql_query("SELECT Number_of_Claims_Past_5_Years FROM home_insurance WHERE Has_Pool ='Yes' AND Has_Trampoline ='No' ", conn)
num_rows_p_no_t = len(df_p_no_t)
claims_p_no_t = df_p_no_t.sum().values[0]
ave_claims_p_no_t = round(claims_p_no_t/num_rows_p_no_t, 2)
print("The average number of claims, given that the house has a pool but doesn't have a trampoline, is", ave_claims_p_no_t)
print("\n")

#c).
df_t_no_p = pd.read_sql_query("SELECT Number_of_Claims_Past_5_Years FROM home_insurance WHERE Has_Pool ='No' AND Has_Trampoline ='Yes' ", conn)
num_rows_t_no_p = len(df_t_no_p)
claims_t_no_p = df_t_no_p.sum().values[0] 
ave_claims_t_no_p = round(claims_t_no_p/num_rows_t_no_p, 2)
print("The average number of claims, given that the house has a trampoline but doesn't have a pool, is", ave_claims_t_no_p)
print("\n")

#d).
df_pt = pd.read_sql_query("SELECT Number_of_Claims_Past_5_Years FROM home_insurance WHERE Has_Pool ='Yes' AND Has_Trampoline ='Yes' ", conn)
num_rows_pt = len(df_pt)
claims_pt = df_pt.sum().values[0] 
ave_claims_pt = round(claims_pt/num_rows_pt, 2)
print("The average number of claims, given that the house has both a pool and a trampoline, is", ave_claims_pt)
print("\n")




sns.barplot(data=df, x='Number_of_Claims_Past_5_Years', y='Annual_Premium', errorbar = "sd")
plt.title("Correlation Between Number of Past Claims and Policy Premium")
plt.ylabel("Annual Premium")
plt.xlabel("Claims in the Past 5 Years")
plt.show()
print("\n")


sns.barplot(data=df, x='Policy_Holder_Age', y='Home_Value', errorbar = "sd")
plt.title("Correlation Between Policy Holder's Age and Home Value")
plt.ylabel("Home Value")
plt.xlabel("Age")
plt.show()
print("\n")


