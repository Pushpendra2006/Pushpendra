#SENTIMENT ANALYSIS WITH NLP
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import string                      
# Load the dataset
data=pd.read_csv('C:/Users/pushp/Downloads/output.csv')

# Preprocess text function 
def preprocess_text(text): 
    text=text.lower()
    text=text.translate(str.maketrans('', '', string.punctuation))   
    stop_words=set(('english'))
    text=' '.join(word for word in text.split() if word not in stop_words) 
    return text 

# Apply preprocessing  
data['Review']=data['Review'].apply(preprocess_text) 
                                                                               
# Split the data
x=data['Review'] 
y=data['Liked']                                                                  
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)  

# Vectorize text using TF-IDF
tfidf_vectorizer=TfidfVectorizer(max_features=5000)
x_train_tfidf=tfidf_vectorizer.fit_transform(x_train)    
x_test_tfidf=tfidf_vectorizer.transform(x_test) 
# Train Logistic Regression model
lor=LogisticRegression()
lor.fit(x_train_tfidf,y_train)

# Evaluate the model
y_pred=lor.predict(x_test_tfidf)
accuracy=accuracy_score(y_test, y_pred)*100
print(accuracy,"%") 
print(classification_report(y_test, y_pred)) 

#Training on XGBoost Classifier
xgc=XGBClassifier()
xgc.fit(x_train_tfidf,y_train)
y_pred1=xgc.predict(x_test_tfidf)
b=accuracy_score(y_test,y_pred1)*100
print(b,"%")