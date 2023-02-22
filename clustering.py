# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 23:27:44 2023

@author: lenovo
"""

               ***HIERARCHICAL CLUSTERING***


1]Problem:=(crime_data.csv)

BUSINESS OBJECTIVE:-Perform the clusters for Crime data using HIERARCHICAL CLUSTERING.





#Importing the Necessary Liabrary
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pylab as plt


#Loading the Dataset
df=pd.read_csv('C:/Users/lenovo/OneDrive/Documents/EXCLER ASSIGNMENTS/Clustering/crime_data.csv')
#EDA
df.info()
df.describe()
df=df.rename(columns={'Murder':'murder','Assault':'assault','UrbanPop':'urbanpop','Rape':'rape'})
df.drop(['Unnamed: 0'],axis=1,inplace=True)
df.shape
df.tail()
df.head()

#Normalization Function

def norm_func(i):
    x = (i - i.min()) /(i.max() - i.min())
    return(x)
#Normalized the DataFrame(df)
df1=norm_func(df.iloc[:,:])
df1.describe()

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z=linkage(df1,method='complete',metric='euclidean')

#Dendrogram           

plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram

from sklearn.cluster import AgglomerativeClustering
h=AgglomerativeClustering(n_clusters=5,linkage='complete').fit(df1)
h.labels_

cluster_labels=pd.Series(h.labels_)

df1['clust'] =cluster_labels # creating a new column and assigning it to new column 

#Rearrange the attributes,in which 1st columns is of clust attributes
df1=df1.iloc[:,[4,0,1,2,3]]
df1.head()

# Aggregate mean of each cluster
df1.iloc[:,:].groupby(df1.clust).mean()

# creating a csv file 
df1.to_csv("crime.csv", encoding = "utf-8")







2]Problem:-  EastWestAirlines.csv

    
BUSINESS OBJECTIVE:-Perform clustering (Hierarchical) for the airlines data to obtain optimum number of clusters.    




#Loading the Dataset

df=pd.read_excel('C:/Users/lenovo/OneDrive/Documents/EastWestAirlines.xlsx',sheet_name='data')

#EDA
df.info()
df.shape
df.head()
df.tail()
df.describe()

#Normalization

def norm_func(i):
    x = (i - i.min()) / (i.max() - i.min())
    return(x)
#Normalized the DataFrame(df)
df1=norm_func(df.iloc[:,:])
df1.describe()

#for creating Dendrogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z=linkage(df1,method='complete',metric='euclidean')

#Dendrogram
plt.figure(figsize=(15,8));plt.title('hierarchy clustering dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z,
               leaf_rotation=0,
               leaf_font_size=15,
)
plt.show()


#Now applying the Agglomerative Clustering to choose 5 clusters.
from sklearn.cluster import AgglomerativeClustering 
h=AgglomerativeClustering(n_clusters=5,linkage='complete').fit(df1)
h.labels_

cluster_labels=pd.Series(h.labels_)

#Creating a new column and assign it to this new column.
df['clust']=cluster_labels


#Rearrange the features,and get Clust features as 1st.
df=df.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]
df.head()

# Aggregate mean of each cluster
df.iloc[:, :].groupby(df.clust).mean()

#Creatung a csv file
df.to_csv('eastwest.csv',encoding='utf-8')
import os
os.getcwd()



           ***K-MEANS CLUSTERING***


1]Problem:-crime.csv

BUSINESS OBJECTIVE:-Perform the clusters for Crime data using K-Means CLUSTERING.



#Importing the Necessary Liabrary
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pylab as plt


#Loading the Dataset
df=pd.read_csv('C:/Users/lenovo/OneDrive/Documents/EXCLER ASSIGNMENTS/Clustering/crime_data.csv')

#EDA
df.info()
df.shape
df.head()
df.tail()
df.describe()
#Remove the unwanted column
df1=df.drop(['Unnamed: 0'],axis=1)
df1.head()

#Normalization

def norm_func(i):
    x = (i - i.min()) / (i.max() - i.min())
    return (x)

#Normalized the dataframe(df)
df1=norm_func(df1.iloc[:,:])
df1.describe()


#Scree Plot or elbow curve
TWSS=[]
k = list(range(2,9))

for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df1)
    TWSS.append(kmeans.inertia_)
    
TWSS
 
#Scree PLot
plt.plot(k,TWSS,'ro-');plt.title('scree plot');plt.xlabel('no of clusters');plt.ylabel('total within sum of square')   

#Select 4 clusters from above scree plot ,which is optimum no of clusters.
model=KMeans(n_clusters=4)
model.fit(df1)

model.labels_# getting the labels of clusters assigned to each row 
df['clust']=pd.Series(model.labels_)# creating a  new column and assigning it to new column 
df.head()


#Put the 'clust' features at 1st position in given datset.
df=df.iloc[:,[5,0,1,2,3,4]]
df.head()

df.loc[]
#Creatng a CSV file
df.to_csv('crimekmeans.csv',encoding='utf-8')
import os
os.getcwd()






2]Problem:-  EastWestAirlines.csv

    
BUSINESS OBJECTIVE:-Perform clustering (K-Means) for the airlines data to obtain optimum number of clusters.    


from sklearn.cluster import KMeans

#Loading the dataset
df=pd.read_excel('C:/Users/lenovo/OneDrive/Documents/EXCLER ASSIGNMENTS/Clustering/EastWestAirlines.xlsx',sheet_name='data')

#EDA
df.info()
df.shape
df.head()
df.tail()
df.describe()

#Normalization

def norm_func(i):
    x = (i - i.min()) / (i.max() - i.min())
    return(x)
    
#Normalizing the DataFrame(df)
df1=norm_func(df.iloc[:,:])    
df1.describe()

#Scree Plot/Elbow Curve

TWSS=[]
k = list(range(2,9))

for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df1)
    TWSS.append(kmeans.inertia_)
    
TWSS    

#Plot Scree plot
plt.plot(k,TWSS,'ro-');plt.title('scree plot');plt.xlabel('no of clusters');plt.ylabel('total within sum of clusters')

#Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 5)
model.fit(df1)

model.labels_# getting the labels of clusters assigned to each row 
df['clust']=pd.Series(model.labels_)# creating a  new column and assigning it to new column 

df.head()

#Put the 'clust' features at 1st position in given datset.
df=df.iloc[:,[12,11,10,9,8,7,6,5,4,3,2,1,0]]

#Creating a CSV file
df.to_csv('airlineskmean.csv',encoding='utf-8')
import os
os.getcwd()


            ***DBSCAN CLUSTERING***
            
                     
            
1]Problem:-crime.csv

    
BUSINESS OBJECTIVE:-Perform the clusters for Crime data using DBSCAN CLUSTERING.
            
        
#Importing the Necessary Liabrary            
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pylab as plt


#Loading the dataset
df=pd.read_csv('C:/Users/lenovo/OneDrive/Documents/EXCLER ASSIGNMENTS/Clustering/crime_data.csv')

#EDA
df.info()
df.head()            
df.tail()
df.shape
df.describe()

#Removing Unwanted Column.
df1=df.iloc[:,1:4]
df1.values

#Standardized the data
s=StandardScaler()
df1=s.fit_transform(df1)

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters
dbscan=DBSCAN(eps=2,min_samples=5)
dbscan.fit(df1)

dbscan.labels_# getting the labels of clusters assigned to each row 

df['clust']=pd.Series(dbscan.labels_)# creating a  new column and assigning it to new column 


#Put the 'clust' features at 1st position in given datset.
df=df.iloc[:,[5,4,3,2,1,0]]

#Creating a CSV file
df.to_csv("dbscancrime.csv", encoding = "utf-8")

import os
os.getcwd()






2]Problem:-  EastWestAirlines.csv

    
BUSINESS OBJECTIVE:-Perform clustering (DBSCAN) for the airlines data to obtain optimum number of clusters.    



    
#Loading the Dataset
df=pd.read_excel('C:/Users/lenovo/OneDrive/Documents/EXCLER ASSIGNMENTS/Clustering/EastWestAirlines.xlsx',sheet_name='data')

#EDA
df.info()
df.shape
df.head()
df.tail()
df.describe()

#Standardized the data
s=StandardScaler()
df1=s.fit_transform(df)

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters
dbscan=DBSCAN(eps=2,min_samples=10)
dbscan.fit(df1)

dbscan.labels_# getting the labels of clusters assigned to each row 

df['clust']=pd.Series(dbscan.labels_)# creating a  new column and assigning it to new column 

#Put the 'clust' features at 1st position in given datset.
df=df.iloc[:,[12,11,10,9,8,7,6,5,4,3,2,1,0]]

#creating the CSV file
df.to_csv("dbscanairlines.csv", encoding = "utf-8")

import os
os.getcwd()