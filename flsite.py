import pickle
import numpy as np
from flask import Flask, render_template, request

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)



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
        return render_template('Lab2.html', prediction=pred)

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
        return render_template('Lab3.html', prediction=pred)

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
        return render_template('Lab1.html', prediction=pred)

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
        return render_template('lab4.html', prediction=pred)

if __name__ == "__main__":
    app.run(debug=True)