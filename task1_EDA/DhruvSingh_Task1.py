import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. load datbase 
df = pd.read_csv('Passengers.csv')

# 2. Data overview
print("First 5 rows of the dataset:")
print(df.head())

# 3. Basic EDA: Count of Survived vs Not Survived
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.show()

# 4. Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())