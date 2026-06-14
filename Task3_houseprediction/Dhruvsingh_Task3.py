import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. Data load 
df = pd.read_csv('Housing.csv')

# 2. Features (X) aur Target (y) 
X = df[['Area', 'Bedrooms', 'Bathrooms']]
y = df['Price']

# 3. Model train 
model = LinearRegression()
model.fit(X, y)

# 4. Prediction check
print("Model trained successfully!")
print("Predicted price for a 3000 sqft house:", model.predict([[3000, 4, 3]]))