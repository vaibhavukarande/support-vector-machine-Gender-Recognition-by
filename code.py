# --------------
# Loading the Necessary Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
df=pd.read_csv(path)

# Check the correlation between each feature and check for null values
#print(df.corr())

# Check for null values
#print(df.isnull().sum())
# Print total no of labels also print number of Male and Female labels
#print("Total Labels",df.shape[0])
#print("Total Male",df[df['label']=='male'].shape[0])
#print("Total Female",df[df['label']=='female'].shape[0])
# Label Encode target variable
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
le=LabelEncoder()
y=le.fit_transform(y)
#print(y)
# Scale all the independent features and split the dataset into training and testing set.
scale=StandardScaler()
X=pd.DataFrame(scale.fit_transform(X))
#print(X.head(n=2))

#Split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=100)
# Build model with SVC classifier keeping default Linear kernel and calculate accuracy score.
model_1=SVC(kernel='linear')
model_1.fit(X_train,y_train)
y_pred=model_1.predict(X_test)
print("Accuracy Score",accuracy_score(y_test,y_pred))
# Build SVC classifier model with polynomial kernel and calculate accuracy score


# Build SVM model with rbf kernel.


#  Remove Correlated Features.


# Split the newly created data frame into train and test set, scale the features and apply SVM model with rbf kernel to newly created dataframe


# Do Hyperparameter Tuning using GridSearchCV and evaluate the model on test data.





