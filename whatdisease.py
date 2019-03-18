from flask import Flask, jsonify, request

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# training
data = pd.read_csv('datasets/dataset_cleaned.csv', encoding="ISO-8859-1")
df_pivoted = pd.read_csv('datasets/df_pivoted.csv', encoding="ISO-8859-1")
columns = df_pivoted.columns
columns = columns[2:]
x = df_pivoted[columns]
y = df_pivoted['Source']
dct_clf = DecisionTreeClassifier()
dct_clf = dct_clf.fit(x, y)

symptoms_list = list(data['Target'].unique())[:404]

app = Flask(__name__)

@app.route('/whatdisease/api/v1.0/symptoms/', methods=['GET'])
def get_symptoms():
    print(len(symptoms_list))
    return jsonify({'symptoms': symptoms_list})


@app.route('/whatdisease/api/v1.0/disease', methods=['GET'])
def get_disease():
    symptoms_given = [request.args[str(i)] for i in range(len(request.args))]
    x_ = np.zeros((1, 404))
    for s in symptoms_given:
        try:
            i = symptoms_list.index(s)
        except:
            return jsonify({'error': 'No matching symptom :('})
        x_[0][i] = 1

    disease = dct_clf.predict(x_)[0]
    return jsonify({'disease': disease})


if __name__ == "__main__":
    print("Starting app...")
    app.run(debug=True, port=5000)
