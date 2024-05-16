import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Load the dataset from Excel file
dataset = pd.read_excel('DataSet.xlsx')

# Преобразование категориальной переменной 'Успешность' в числовой формат
label_encoder = LabelEncoder()
dataset["Успешность"] = label_encoder.fit_transform(dataset["Успешность"])

# Display the first few rows of the dataset to understand its structure
print(dataset.head())

# Assuming the last column is the target variable
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
pickle.dump(linear_model, open('linear_model.pkl', 'wb'))

# Logistic Regression
# Создание экземпляра StandardScaler
scaler = StandardScaler()

# Масштабирование признаков
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение модели логистической регрессии на масштабированных данных
logistic_model = LogisticRegression()
logistic_model.fit(X_train_scaled, y_train)

# Сохранение модели
pickle.dump(logistic_model, open('logistic_model.pkl', 'wb'))

# KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
pickle.dump(knn_model, open('knn_model.pkl', 'wb'))

# Decision Tree
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)
pickle.dump(tree_model, open('tree_model.pkl', 'wb'))
