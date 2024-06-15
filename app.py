from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import io
import base64

app = Flask(__name__)

Titanic_data = pd.read_csv("Titanic-Dataset.csv")

Titanic_data.drop(["Cabin"], axis=1, inplace=True)
Titanic_data["Embarked"] = Titanic_data["Embarked"].fillna("S")

mean_age = Titanic_data["Age"].mean()
std_age = Titanic_data["Age"].std()
random_ages = np.random.randint(mean_age - std_age, mean_age + std_age, Titanic_data["Age"].isnull().sum())
Titanic_data.loc[Titanic_data["Age"].isnull(), "Age"] = random_ages
Titanic_data["Sex"] = Titanic_data["Sex"].map({"male": 0, "female": 1})
boarding_point = pd.get_dummies(Titanic_data["Embarked"], drop_first=True)
Titanic_data.drop(["Name", "Ticket", "PassengerId", "Embarked"], axis=1, inplace=True)
Titanic_data = pd.concat([Titanic_data, boarding_point], axis=1)

X = Titanic_data.drop("Survived", axis=1)
Y = Titanic_data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analysis')
def analysis():
    predicted_res = model.predict(X_test)
    accuracy = accuracy_score(y_test, predicted_res)
    conf_matrix = confusion_matrix(y_test, predicted_res)
    class_report = classification_report(y_test, predicted_res)

    plt.figure(figsize=(8, 6))
    survived_counts = Titanic_data['Survived'].value_counts()
    plt.pie(survived_counts, labels=['Not Survived', 'Survived'], autopct='%1.1f%%', startangle=140, colors=['red', 'green'])
    plt.title('Survived vs Not Survived')
    plt.axis('equal')
    img1 = io.BytesIO()
    plt.savefig(img1, format='png')
    img1.seek(0)
    plot_url1 = base64.b64encode(img1.getvalue()).decode()

    plt.figure(figsize=(8, 6))
    class_counts = Titanic_data['Pclass'].value_counts()
    plt.pie(class_counts, labels=['Class 1', 'Class 2', 'Class 3'], autopct='%1.1f%%', startangle=140, colors=['gold', 'lightblue', 'lightgreen'])
    plt.title('Distribution of Class')
    plt.axis('equal')
    img2 = io.BytesIO()
    plt.savefig(img2, format='png')
    img2.seek(0)
    plot_url2 = base64.b64encode(img2.getvalue()).decode()

    return render_template('index.html', accuracy=accuracy, conf_matrix=conf_matrix, class_report=class_report, plot_url1=plot_url1, plot_url2=plot_url2)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        passenger_class = int(request.form['pclass'])
        sex = int(request.form['sex'])
        age = float(request.form['age'])
        siblings_spouses = int(request.form['siblings_spouses'])
        parents_children = int(request.form['parents_children'])
        fare = float(request.form['fare'])
        embarked_Q = int(request.form.get('embarked_Q', 0))
        embarked_S = int(request.form.get('embarked_S', 0))

        input_data = [[passenger_class, sex, age, siblings_spouses, parents_children, fare, embarked_Q, embarked_S]]
        prediction = model.predict(input_data)

        result = 'Survived' if prediction[0] == 1 else 'Not Survived'

        return render_template('index.html', prediction_result=result)
    
if __name__ == '__main__':
    app.run(debug=True)
