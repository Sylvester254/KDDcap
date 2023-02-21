#!/usr/bin/python3
"""_summary_
Binary classification of network connections
as normal or malicious using a Random Forest algorithm.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import cross_val_predict, cross_val_score
# from sklearn.metrics import classification_report



data = pd.read_csv('kddcup.data_10_percent_corrected',
                   delimiter=',', header=None)
# Define column names
column_names = ['duration', 'protocol_type', 'service', 'flag',
                'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
                'urgent', 'hot', 'num_failed_logins', 'logged_in',
                'num_compromised', 'root_shell', 'su_attempted', 'num_root',
                'num_file_creations', 'num_shells', 'num_access_files',
                'num_outbound_cmds', 'is_host_login', 'is_guest_login',
                'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
                'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
                'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                'dst_host_srv_count', 'dst_host_same_srv_rate',
                'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                'dst_host_srv_rerror_rate', 'label']

# Assign the column names to the DataFrame and select relevant columns
data.columns = column_names
relevant_columns = ["protocol_type", "service", "src_bytes",
                    "dst_bytes", "count", "srv_count", "same_srv_rate",
                    "diff_srv_rate", "dst_host_count", "dst_host_srv_count",
                    "label"]
relevant_data = data[relevant_columns]
# print(data.head(10))

"""_summary_
    Encoding categorical variables and scaling the numerical features
"""
encoded_data = pd.get_dummies(relevant_data, columns=['protocol_type', 'service'])
scaler = StandardScaler()
scaled_data = encoded_data.copy()
scaled_data[['src_bytes', 'dst_bytes']] = scaler.fit_transform(
    encoded_data[['src_bytes', 'dst_bytes']])
# print(scaled_data.head(10))


"""_summary_
    -Split the data into input features (X) and target variable (y)
    -Split the data into training and testing sets
    -Create a Random Forest classifier instance
    with 100 trees and a random seed of 42
    -Train the model using the training set
    -Make predictions on the test set

"""
X = scaled_data.drop('label', axis=1)
y = scaled_data['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# print("Data types of X_train:\n", X_train.dtypes)
# print("Data types of X_test:\n", X_test.dtypes)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# predicting the test results
y_pred = rf_classifier.predict(X_test)


"""_summary_
    Evaluate the performance of the model
"""
print('Accuracy: ', accuracy_score(y_test, y_pred))

print('precision: ', precision_score(y_test,
                                    y_pred, average='weighted'))
print('recall:', recall_score(y_test, y_pred, average='weighted'))

print('normal_precision:', precision_score(y_test, y_pred,
                                           labels=['normal'],
                                           average='weighted'))
print('normal_recall:', recall_score(y_test, y_pred,
                                     labels=['normal'], average='weighted'))


#Creating the Confusion matrix  
cm= confusion_matrix(y_test, y_pred) 
print(cm)
