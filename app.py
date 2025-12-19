#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install flask pandas')
get_ipython().system('pip install nest_asyncio')
import nest_asyncio
nest_asyncio.apply()



# In[4]:


from flask import Flask, render_template, request
import pandas as pd
import random


# In[ ]:


from flask import Flask, render_template, request
import pandas as pd
import random

app = Flask(__name__)

df = pd.read_csv('imdb_clean.csv')

@app.route("/", methods=["GET", "POST"])
def home():
    movies = []
    
    if request.method == "POST":
        genre = request.form["genre"].lower()
        filtered = df[df["genre"].str.contains(genre, case=False, na=False)]
        
        if not filtered.empty:
            # Pick up to 5 random movies
            movies = random.sample(filtered["title"].tolist(), min(5, len(filtered)))
        else:
            movies = ["No movie found for this genre"]
    
    return render_template("home.html", movies=movies)

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False, port=5000)


# In[ ]:


df.head()


# In[ ]:




