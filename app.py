from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the voting classifier model
filename = 'model/voting_diabetes.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__, template_folder="templates")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        
        data = np.array([[glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)

        if my_prediction[0] == 0:
            output = "No Diabetes"
        else:
            output = "Diabetes"

    return render_template('index.html', prediction_text="Result: {}".format(output))

def main():
    app.run()

if __name__ == '__main__':
    main()
