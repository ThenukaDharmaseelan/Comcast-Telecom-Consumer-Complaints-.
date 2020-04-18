#!/usr/bin/env python
# coding: utf-8

# In[1]:


cd


# In[2]:


cd Desktop


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


df = pd.read_csv("Comcast_telecom_complaints_data.csv")


# In[5]:


df.head(5)


# In[6]:


df["date_index"] = df["Date_month_year"] + " " + df["Time"]


# In[7]:


df["date_index"] = pd.to_datetime(df["date_index"])
df["Date_month_year"] = pd.to_datetime(df["Date_month_year"])


# In[8]:


df.dtypes


# In[9]:


df.head()


# In[10]:


df = df.set_index(df["date_index"])


# In[11]:


df.head(5)


# In[12]:


df["Date_month_year"].value_counts()


# In[13]:



df["Date_month_year"].value_counts().plot();


# In[14]:



f = df.groupby(pd.Grouper(freq="M")).size()


# In[15]:


f.head()


# In[16]:


#monthly chart
df.groupby(pd.Grouper(freq="M")).size().plot()


# In[17]:


f = df.groupby(pd.Grouper(freq="D")).size()


# In[18]:


f.head()


# In[19]:


#Daily chart
df.groupby(pd.Grouper(freq="D")).size().plot()


# In[20]:


# frequency of complaint types.
df.groupby(["Customer Complaint"]).size().sort_values(ascending=False).to_frame().reset_index().rename({0: "Count"}, axis=1)


# In[21]:


# maximum
df.groupby(["Customer Complaint"]).size().sort_values(ascending=False).to_frame().reset_index().rename({0: "Count"}, axis=1).max()


# In[22]:



df["newStatus"] = ["Open" if Status=="Open" or Status=="Pending" else "Closed" for Status in df["Status"]]


# In[23]:


df.head()


# In[24]:



df.groupby(["State"]).size().sort_values(ascending=False).to_frame().reset_index().rename({0: "Count"}, axis=1)


# In[25]:


Status_complaints = df.groupby(["State","newStatus"]).size().unstack().fillna(0)
Status_complaints


# In[26]:


Status_complaints.plot(kind="barh", figsize=(30,50), stacked=True)


# In[27]:


df.groupby(["State"]).size().sort_values(ascending=False).to_frame().reset_index().rename({0: "Count"}, axis=1).max()


# In[28]:



df.groupby(["State","newStatus"]).size().unstack().fillna(0).max()


# In[29]:


pip install wordcloud


# In[30]:


pip install nltk


# In[31]:



from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string


# In[32]:


import nltk
nltk.download('stopwords')


# In[33]:


stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


# In[34]:



def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = "".join([ch for ch in stop_free if ch not in exclude])
    normalised = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalised


# In[35]:


import nltk
nltk.download('wordnet')


# In[36]:


doc_complete = df["Customer Complaint"].tolist()
doc_clean = [clean(doc).split() for doc in doc_complete]


# In[37]:


pip install gensim


# In[38]:


import gensim
from gensim import corpora


# In[39]:



dictionary = corpora.Dictionary(doc_clean)
print(dictionary)


# In[40]:


doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
doc_term_matrix


# In[41]:



from gensim.models import LdaModel


# In[42]:


Num_Topic = 9
ldamodel = LdaModel(doc_term_matrix, num_topics= Num_Topic, id2word= dictionary, passes= 30)


# In[43]:


topics = ldamodel.show_topics()
for topic in topics:
    print(topic)
    print()


# In[44]:


word_dict = {}
for i in range(Num_Topic):
    words = ldamodel.show_topic(i,topn =30)
    word_dict["Topic # " + "{}".format(i)] = [i[0] for i in words]


# In[45]:



pd.DataFrame(word_dict)


# In[ ]:




