import pickle
import numpy as np
from flask import Flask, render_template, request
from flask import jsonify
import pandas as pd
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

#dataset = pd.read_excel('DataSet.xlsx')

def get_metric(my_model, array):
    labelencoder = LabelEncoder()
    dataset = pd.read_excel('DataSet.xlsx')

    X = dataset.drop(["Зарплата"], axis=1)
    Y = dataset["Зарплата"]

    X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X, Y, test_size=0.2, random_state=3)

    train_y = Y_train1.values.reshape(-1, 1)
    train_x = X_train1.values

    # Преобразование данных с помощью LabelEncoder
    train_x[:, 3] = labelencoder.fit_transform(train_x[:, 3])

    # Обучение модели
    my_model.fit(train_x, train_y.ravel())

    # Предсказание
    znach = my_model.predict(train_x)

    # Округление предсказанных значений
    znach = np.round(znach)

    y_true = train_y.ravel()
    y_pred = znach

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # print(f"MSE: {mse}")
    # print(f"MAE: {mae}")
    # print(f"MAPE: {mape}")
    # print(f"R2: {r2}")

    return [mse, mae, mape, r2]



# Load the trained models
loaded_linear_model = pickle.load(open('linear_model.pkl', 'rb'))
loaded_logistic_model = pickle.load(open('logistic_model.pkl', 'rb'))
loaded_knn_model = pickle.load(open('knn_model.pkl', 'rb'))
loaded_tree_model = pickle.load(open('tree_model.pkl', 'rb'))

@app.route("/")
def index():
    menu = [
        {
            "name": "name",
            "url": "/"
        },
        {
            "name": "linear_regression",
            "url": "/linear_regression"
        },
        {
            "name": "logistic_regression",
            "url": "/logistic_regression"
        },
        {
            "name": "knn 3",
            "url": "/knn"
        },
        {
            "name": "decision_tree",
            "url": "./decision_tree"
        }
    ]
    return render_template('index.html', menu=menu )

@app.route("/linear_regression", methods=['POST', 'GET'])
def linear_regression():
    if request.method == 'GET':
        return render_template('Lab2.html')
    if request.method == 'POST':
        # Assuming the form input names are 'feature1', 'feature2', 'feature3', 'feature4'
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),
                           float(request.form['list4'])]])
        pred = loaded_linear_model.predict(X_new)

        my_model = LinearRegression()
        metric = get_metric(my_model, X_new)

        return render_template('Lab2.html', prediction=pred, metric=metric)

@app.route("/logistic_regression", methods=['POST', 'GET'])
def logistic_regression():
    if request.method == 'GET':
        return render_template('Lab3.html')
    if request.method == 'POST':
        # Assuming the form input names are 'feature1', 'feature2', 'feature3', 'feature4'
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),
                           float(request.form['list4'])]])
        pred = loaded_logistic_model.predict(X_new)

        my_model = LogisticRegression()
        metric = get_metric(my_model, X_new)

        return render_template('Lab3.html', prediction=pred, metric=metric)

@app.route("/knn", methods=['POST', 'GET'])
def knn():
    if request.method == 'GET':
        return render_template('Lab1.html')
    if request.method == 'POST':
        # Assuming the form input names are 'feature1', 'feature2', 'feature3', 'feature4'
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),
                           float(request.form['list4'])]])
        types = ["Доходяга", "Обычный", "Успешный"]
        pred = types[loaded_knn_model.predict(X_new)[0]]

        my_model = LinearRegression()
        metric = get_metric(my_model, X_new)

        return render_template('Lab1.html', prediction=pred, metric=metric)

@app.route("/decision_tree", methods=['POST', 'GET'])
def decision_tree():
    if request.method == 'GET':
        return render_template('lab4.html')
    if request.method == 'POST':
        # Assuming the form input names are 'feature1', 'feature2', 'feature3', 'feature4'
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),
                           float(request.form['list4'])]])
        pred = loaded_tree_model.predict(X_new)

        my_model = DecisionTreeClassifier()
        metric = get_metric(my_model, X_new)

        return render_template('lab4.html', prediction=pred, metric=metric)

# @app.route('/api', methods=['get'])
# def get_sort():
#     X_new = np.array([[float(request.args.get('sepal_length')),
#                        float(request.args.get('sepal_width')),
#                        float(request.args.get('petal_length')),
#                        float(request.args.get('petal_width'))]])
#     pred = loaded_knn_model.predict(X_new)
#
#     return jsonify(sort=pred[0])


if __name__ == "__main__":
    app.run(debug=True)