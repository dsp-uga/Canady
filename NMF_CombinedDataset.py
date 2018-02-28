
# Importing Spark
import pyspark
from pyspark import SparkConf,SparkContext
import thunder as td
from extraction import NMF
from os import listdir
import json

# Configuring Spark

conf = SparkConf().setAppName("neuronSegmentation")
conf = (conf.setMaster('local[*]')
        .set('spark.executor.memory', '4G')
        .set('spark.driver.memory', '10G'))
sc = SparkContext(conf=conf)

# # Getting List of All test data

path = "/home/vyom/UGA/DSP/Project3/data/test/"
onlyfiles = [f for f in listdir(path)]

# # Creating list of thunder image Vectors of all dataset

data=list()
for i in range (len(onlyfiles)):
    data.append(td.images.fromtif(path=path+onlyfiles[i]+'/images',engine=sc,ext="tiff"))
print("DATA READ!")

# # Creating NMF model

algorithm = NMF(k=10, max_iter=20, percentile=95, overlap=0.1)

# # Fitting models for each dataset

model=list()
for i in range(len(data)):
    model.append(algorithm.fit(data[i], chunk_size=(50,50)))

merged=list()
for i in range(len(model)):
    merged.append(model[i].merge(overlap=0.1))

# Saving region coordinates as model:

for i in range(len(merged)):
    coordinates = [{'coordinates': x.coordinates.tolist()} for x in merged[i].regions]
    jsonString = {'dataset': onlyfiles[i].replace("neurofinder.",""), 'regions': coordinates}
    with open('output' + onlyfiles[i].replace("neurofinder.","")  +'.json', 'w') as f:
        f.write(json.dumps(jsonString))

