import pandas as pd
from sklearn.linear_model import LinearRegression

data=pd.read_csv("house_price_dataset.csv")
print(data)

X=data[['Area_sqft','Bedrooms','Bathrooms']]
print(X)

Y=data.Price
model=LinearRegression()
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

model.fit(X_train, Y_train)

print("Accuracy:", model.score(X_test, Y_test) * 100, "%")

model.predict([[1000,2,2]])
