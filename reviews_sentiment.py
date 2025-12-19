#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #TF-IDF (Term Frequency-Inverse Document Frequency) - tells how important a word is 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import _stop_words

stop = _stop_words.ENGLISH_STOP_WORDS

df = pd.read_csv("IMDB Dataset.csv")   
print(df.head())

def clean_text(text):
    text = text.lower()                                  
    text = re.sub(r"[^a-z ]", "", text)                  
    text = " ".join([word for word in text.split() 
                     if word not in stop])              
    return text

df["cleaned_review"] = df["review"].apply(clean_text)

df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

X_text = df["cleaned_review"]
y = df["sentiment"]

tfidf = TfidfVectorizer(max_features=5000) 
X = tfidf.fit_transform(X_text).toarray()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Model Accuracy:", acc)


def predict_sentiment(review):
    cleaned = clean_text(review)
    vector = tfidf.transform([cleaned]).toarray()
    prediction = model.predict(vector)[0]
    return "Positive" if prediction == 1 else "Negative"

print("Sentiment Predictor Ready")

while True:
    review = input("Enter a movie review (or 'exit' to stop): ")
    if review.lower() == "exit":
        break
    print("Sentiment:", predict_sentiment(review))
    print()


# In[ ]:


pip uninstall keras keras-nightly keras-preprocessing -y
pip uninstall tensorflow -y
pip uninstall protobuf -y



# In[ ]:


pip install tf-keras
pip install tensorflow==2.15
pip install protobuf==3.20.*
pip install transformers==4.36
pip install huggingface_hub


# In[2]:


from transformers import pipeline

classifier = pipeline("sentiment-analysis")
print(classifier("I love using BERT models, they are amazing!"))


# In[4]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


# In[6]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = "The product was really good!"

inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)

probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

print("Positive:", float(probs[0][1]))
print("Negative:", float(probs[0][0]))


# In[8]:


pip install transformers tensorflow


# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf


df = pd.read_csv("IMDB Dataset.csv")
df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

X = df["review"].astype(str).tolist()
y = df["sentiment"].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2
)

def tokenize(batch):
    return tokenizer(
        batch,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="tf"
    )

train_encodings = tokenize(X_train)
test_encodings = tokenize(X_test)


train_dataset = tf.data.Dataset.from_tensor_slices(
    (dict(train_encodings), y_train)
).batch(16)

test_dataset = tf.data.Dataset.from_tensor_slices(
    (dict(test_encodings), y_test)
).batch(16)


model.compile(
    optimizer=Adam(learning_rate=2e-5),
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=2,
    callbacks=[EarlyStopping(monitor='val_loss', patience=1)]
)

def predict_sentiment(review):
    tokens = tokenize([review])
    output = model.predict(dict(tokens))
    logits = output.logits
    prediction = tf.math.argmax(logits, axis=1).numpy()[0]
    return "Positive" if prediction == 1 else "Negative"

print("\nBERT Sentiment Predictor Ready\n")

# -------------------------------------------------------
# 8. User Input Loop (same as your TF-IDF program)
# -------------------------------------------------------
while True:
    review = input("Enter a movie review or 'exit': ")
    if review.lower() == "exit":
        break

    print("Sentiment:", predict_sentiment(review))
    print()


# In[ ]:




