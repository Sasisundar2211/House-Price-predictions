from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Load dataset
X, y = load_iris(return_X_y=True)

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Create KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

#Predict & evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred)) 