
# coding: utf-8

# In[1]:


#use this if you want to use and initialize sprak, sparkcontext
import pyspark
from pyspark import SparkConf,SparkContext

import thunder as td
from extraction import NMF
from os import listdir
import json


# In[2]:


#use engine=sc as formatted below to use spark while reading th data


# In[3]:


#set spark config accoring to your loacl machine [master and memory]
conf = SparkConf().setAppName("neuronSegmentation")
conf = (conf.setMaster('local[*]')
        .set('spark.executor.memory', '4G')
        .set('spark.driver.memory', '10G'))
sc = SparkContext(conf=conf)


# # Creating list of thunder image Vectors of all dataset

# In[4]:


#the path of data is hard-code for now. 
#when running as python file , we can take it as an input argument
data = td.images.fromtif(path='/home/hiten/Desktop/neurofinder.03.00/images',engine=sc,ext="tiff")
print("DATA READ!")


# # Creating NMF model

# In[15]:


#create the model and play with various values of k,percentile to get efficient results
algorithm = NMF(k=5, max_iter=30, percentile=95, overlap=0.1)


# # Fitting models for each dataset

# In[16]:


#fit our data in the model
model = algorithm.fit(data, chunk_size=(50,50))


# In[17]:


#fixing overlapping pixels
merged = model.merge(overlap=0.1)


# In[18]:


#saving cordinates value in a list and passing it to jsonString
coordinates = [{'coordinates': x.coordinates.tolist()} for x in merged.regions]


# In[19]:


jsonString = {'dataset': "03.00.test", 'regions': coordinates}


# In[20]:


#saving the desired format output to submit on AutoLAB
with open('output' + "03.00.test" +'.json', 'w') as f:
    f.write(json.dumps(jsonString))

