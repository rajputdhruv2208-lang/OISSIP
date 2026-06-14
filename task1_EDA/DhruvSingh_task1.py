import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset load karein (ensure karein ki csv file isi folder mein hai)
df = pd.read_csv('retail_sales.csv')

# Data saaf karein
df.dropna(inplace=True) 

# Sales ka graph banayein
plt.figure(figsize=(10,6))
plt.plot(df['Date'], df['Sales'])
plt.title('Retail Sales Trend')
plt.show()

# Category ka graph banayein
sns.barplot(x='Category', y='Sales', data=df)
plt.title('Sales by Category')
plt.show()