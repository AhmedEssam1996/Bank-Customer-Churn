#=====================
#   import library
#====================='

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#==================='
#   import data
#==================='

bank_data=pd.read_csv(r'C:\Users\Elbostan\Desktop\power bi project\Bank+Customer+Churn\Bank_Churn.csv')

#======================='
#   Data preprocessing.
#======================='

info=bank_data.info()
bank_data.isna().sum()
Description=bank_data.describe()
plt.figure(figsize=(10, 6))
sns.heatmap(Description,annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Statistical Summary Heatmap")
plt.show()



#=============================================================='
#                        Data Analysis
#=============================================================='

"""
What attributes are more common among churners than non-churners?
Can churn be predicted using the variables in the data?

What do the overall demographics of the bank's customers look like?

Is there a difference between German, French, and Spanish customers
 in terms of account behavior?

What types of segments exist within the bank's customers?"""



#==============================================
'''       What attributes are more common 
          among churners than non-churners                               '''
#==============================================

data = bank_data.drop(['Surname','Geography','Gender'], axis=1)
correlation =data.corr()["Exited"].sort_values(ascending=False)
print(correlation)

plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("What attributes are more common among churners than non-churners")
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x="Exited", y="Balance", data=bank_data)
plt.title("balance vs Exited")
plt.show()



#====================================================================
"""            What do the overall demographics of the
                      bank's customers look like?                                        """
#====================================================================

"Age Distribution"

plt.figure(figsize=(10,5))
sns.histplot(bank_data["Age"], bins=30, kde=True, color="blue")
plt.title("Age Distribution of Bank Customers")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.show()



"Gender Distribution"

sns.countplot(x="Gender", data=bank_data, palette="pastel")
plt.title("Gender Distribution of Customers")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")
plt.show()


"Salary Distribution"

sns.histplot(bank_data["EstimatedSalary"], bins=30, kde=True, color="green")
plt.title("Estimated Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Number of Customers")
plt.show()


"Tenure Distribution"

sns.countplot(x="Tenure", data=bank_data, palette="coolwarm")
plt.title("Customer Tenure Distribution")
plt.xlabel("Years with the Bank")
plt.ylabel("Number of Customers")
plt.show()


"Credit Score Distribution"

sns.histplot(bank_data["CreditScore"], bins=30, kde=True, color="purple")
plt.title("Credit Score Distribution of Customers")
plt.xlabel("Credit Score")
plt.ylabel("Number of Customers")
plt.show()


#===================================================================
"""      Is there a difference between German, French, 
      and Spanish customers in terms of account behavior?                           """
#===================================================================

sns.countplot(x="Geography", hue="IsActiveMember", data=bank_data, palette="Set2")
plt.title("Account Activity by Country")
plt.xlabel("Country")
plt.ylabel("Number of Customers")
plt.legend(title="Active Member", labels=["Inactive", "Active"])
plt.show()



