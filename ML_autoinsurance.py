import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


######################## PART 1 : DATA PRE-PROCESSING ########################

# Load the data
train_data = pd.read_csv('Data\\train_auto.csv')
test_data = pd.read_csv('Data\\test_auto.csv') 

numerical_features=['KIDSDRIV','AGE','HOMEKIDS','YOJ','TRAVTIME','TIF','CLM_FREQ','MVR_PTS','CAR_AGE']
dollars_features=['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM'] 
binary_features=['PARENT1','MSTATUS','SEX','RED_CAR','REVOKED']
categorical_features=['EDUCATION','JOB','CAR_USE','CAR_TYPE','URBANICITY']

# Pre-processing function : clean data, fill missing values and encode categorical data
def datapreprocess(data):
    
    # Turning dollars amounts into floats
    for i in dollars_features :   
        data[i] = data[i].str.replace('$','').str.replace(',','.')
        data[i] = data[i].astype(float)      
    
    # Filling missing values
    for i in numerical_features+dollars_features :
        if data[i].isnull().sum() != 0 :
            data[i] = data[i].fillna(data[i].median())  #NaN values are replaced by the median value of the feature      
    for i in binary_features+categorical_features :
        if data[i].isnull().sum() != 0 :
            data[i] = data[i].fillna(data[i].mode()[0]) #NaN values are replaced by the most represented value of the feature           
    
    # Binarization for variables yes/no and male/female
    data.replace({"yes":1,"no":0,"Yes":1,"No":0,"z_No":0,"M":0,"z_F":1}, inplace=True)          
    
    # Encoding categorical features    
    for i in categorical_features : 
          labelencoder=LabelEncoder()
          data[i]=labelencoder.fit_transform(data[i])       
    
    # Removing non necessary columns
    data.drop(['TARGET_AMT'],axis='columns',inplace=True) # empty in test dataset
    data.drop(['INDEX'],axis='columns',inplace=True) # the index doesn't impact predictions
        

# Pre-processing training dataset and displaying resulting data
traindatacopy = train_data.copy()
datapreprocess(traindatacopy) 
pd.set_option('display.max_columns', None) # To view all columns at once when looking at a dataframe   
print(traindatacopy.head(10))


######################## PART 2 : MODEL SELECTION #########################

# Splitting traing dataset for validation testing (80% for training and 20% for validation)
Y = traindatacopy.loc[:,'TARGET_FLAG']
X = traindatacopy.drop(['TARGET_FLAG'],axis=1) 
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=1, shuffle=True)

# Model selection : supervised classification methods
models = []
models.append(('XGboost', XGBClassifier()))
models.append(('LogisticRegression', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('DecisionTree', DecisionTreeClassifier()))
models.append(('RandomForest', RandomForestClassifier()))
models.append(('NaiveBayes', GaussianNB()))

# Models evaluation : metrics measured on validation dataset
for i_name, i_model in models:
    i_model.fit(X_train, Y_train)
    Y_pred = i_model.predict(X_validation)
    # Metrics
    accuracy = accuracy_score(Y_validation, Y_pred)
    precision = precision_score(Y_validation, Y_pred)
    f1 = f1_score(Y_validation, Y_pred)
    print('%s: accuracy %f precision %f f1 %f' % (i_name, accuracy, precision, f1))
    
# Best method is Random Forest based on confusion matrix and metrics values


######################## PART 3 : MODEL ASSESMENT ############################
    
# Parameters for Random Forest Classifier
n_estinators_list=np.arange(200, 1000, 200)
for i in n_estinators_list:
    model = RandomForestClassifier(n_estimators=i)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_validation)
    accuracy = accuracy_score(Y_validation, Y_pred)
    print('Random Forest with n_estimators=%f: accuracy %f' % (i, accuracy))

# Metrics for n-estimators=800
precision = precision_score(Y_validation, Y_pred)
f1 = f1_score(Y_validation, Y_pred)
print('Random Forest (n_estimators=800): precision %f f1 %f' % (precision, f1))
# Confusion matrix
confusion = confusion_matrix(Y_validation, Y_pred)
confusion_display = ConfusionMatrixDisplay(confusion).plot()
plt.title('Random Forest : Confusion matrix')


######################## PART 3 : PREDICTIONS ############################

# Test datasets pre-processing
testdatacopy = test_data.copy()
datapreprocess(testdatacopy)
X_test = testdatacopy.drop(['TARGET_FLAG'],axis=1) 

# Model traning with entire training dataset (X, Y)
model = RandomForestClassifier(n_estimators=800)
model.fit(X, Y)

# Prediction
predictions = model.predict(X_test)

# Export to csv file
results_export = pd.DataFrame({'INDEX': test_data['INDEX'],'TARGET_FLAG': predictions})
results_export.to_csv('test_predictions.csv', index=False) 
