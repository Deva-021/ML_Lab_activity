import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv("Chicago_Taxi_Trips.csv")

print(data.head())
print(data.shape)
print(data.isnull().sum())

X = data.iloc[:, :2].values
y = data.iloc[:, 4:5].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(y_pred)
print(y_test)

plt.scatter(X_train[:, 0], y_train, color='blue')
plt.plot(X_train[:, 0], model.predict(X_train), color='red')
plt.title('FARE VS TRIP MILES (Training Set)')
plt.xlabel('Trip Miles')
plt.ylabel('Fare')
plt.show()

plt.scatter(X_test[:, 0], y_test, color='blue')
plt.plot(X_train[:, 0], model.predict(X_train), color='red')
plt.title('FARE VS TRIP MILES (Testing Set)')
plt.xlabel('Trip Miles')
plt.ylabel('Fare')
plt.show()