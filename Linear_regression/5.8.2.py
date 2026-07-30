import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv("body_weight.csv")

print(data.head())
print(data.shape)
print(data.isnull().sum())

data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})

X = data.iloc[:, 1:].values
y = data.iloc[:, :1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(y_pred)
print(y_test)

plt.scatter(X_train[:, 0], y_train, color="blue")
plt.plot(X_train[:, 0], model.predict(X_train), color="red")
plt.title("WEIGHT VS HEIGHT (Training Set)")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.show()

plt.scatter(X_test[:, 0], y_test, color="blue")
plt.plot(X_train[:, 0], model.predict(X_train), color="red")
plt.title("WEIGHT VS HEIGHT (Testing Set)")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.show()