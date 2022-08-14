import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

data = pd.read_json("sarcastable.json", lines=True)


data["is_sarcastic"] = data["is_sarcastic"].map({0: "Not Sarcasm", 1: "Sarcasm"})

#headlines as x and sarcasm as y? is this necessary or haphazard
#its essentially haphazard, but helps with understanding
#no it is not haphazard. train_test_split syntax implies that training data is X and feedback is y
data = data[["headline", "is_sarcastic"]]

x = np.array(data["headline"])
y = np.array(data["is_sarcastic"])

cv = CountVectorizer()
X = cv.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)   

model = BernoulliNB()
model.fit(X_train, y_train)
print("Model Accuracy: ")
print(model.score(X_test,y_test))

user = input("Speak: ")
data = cv.transform([user])
out = model.predict(data)
print(out)
