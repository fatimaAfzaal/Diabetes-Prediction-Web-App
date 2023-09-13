                                                        #for data manipulation
import numpy as np   
import pandas as pd  
import pickle                                                    #for numerical operations                                                    #for visualization
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
                        

df = pd.read_csv('diabetes.csv') #Read a CSV file by given path

df = df.rename(columns={'DiabetesPedigreeFunction':'DPF'})

df.isnull().values.any()

# Data spliting

from sklearn.model_selection import train_test_split
X = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DPF','Age']]
Y = df['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3, random_state=0)


# Create individual classifiers
rf = RandomForestClassifier(n_estimators=200)
lr = LogisticRegression()
svm = SVC(probability=True)

# Create a voting classifier
voting_classifier = VotingClassifier(estimators=[('rf', rf), ('lr', lr), ('svm', svm)], voting='soft')

# Fit the voting classifier on the training data
voting_classifier.fit(X_train, Y_train)

# Evaluate the accuracy of the voting classifier on the test data
accuracy = voting_classifier.score(X_test, Y_test)
print("Accuracy of Voting Classifier:", accuracy)

# Save the voting classifier to a file
filename = 'model/voting_diabetes.pkl'
pickle.dump(voting_classifier, open(filename, 'wb'))
print("SUCCESS")

