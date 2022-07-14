<div align="center">
<h1> Covid19-Literature-Clustering </h1>

<i>With the increase in scientific literature, there was a need for those documents to be organised. Using machine learning techniques, I will organise and visualise the scientific literature on or related to COVID-19, so that papers on similar topics are grouped together. This simplifies the navigation of topics and related papers. I plan to use the well-known CORD-19 dataset to carry out this approach.</i></div>

<p align="center">
<img src="https://github.com/raofida75/Covid19-Literature-Clustering/blob/master/images/Clusters%20TSNE.png" width="1000"/>
</p>

## Table of Contents

1. [Requirements](#requirements)
2. [Dataset](#dataset)     
3. [Methodology](#methodology)  
	- [Data Preparation](#data-preparation) 
	- [Clustering Approach](#clustering-approach) 	
4. [Results](#results) 
5. [Summary](#summary)

## Requirements
- pandas
- numpy
- matplotlib
- seaborn
- tensorflow
- sklearn 
- scipy
- nltk
- re

## Dataset
[Link to the dataset](https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge)

In response to the COVID-19 pandemic, the White House and a coalition of leading research groups have prepared the COVID-19 Open Research Dataset (CORD-19). CORD-19 is a resource of over 1,000,000 scholarly articles, including over 400,000 with full text, about COVID-19, SARS-CoV-2, and related coronaviruses. This freely available dataset is provided to the global research community to apply recent advances in natural language processing and other AI techniques to generate new insights in support of the ongoing fight against this infectious disease. There is a growing urgency for these approaches because of the rapid acceleration in new coronavirus literature, making it difficult for the medical research community to keep up.

## Methodology 

### [Data Preparation](https://github.com/raofida75/Covid19-Literature-Clustering/blob/master/notebook/Data%20Preparation.ipynb)
Following steps have been performed to prepare the data to perform clustering.
- Load metadata
- Get the path of all json files.
- Extract the body text of all the papers from the json files.
- Combine the metadata and the body text.
- Determine the language of each paper and keep only those in English.
- Remove null and duplicate values

### Clustering Approach
[Link to the notebook](https://nbviewer.org/github/raofida75/Covid19-Literature-Clustering/blob/master/notebooks/Covid19%20Literature%20Clustering.ipynb)
#### 1. Preprocess body text
	- Remove Stop words
	- Perform Lematization. 

#### 2. Vectorization
	- Use Tf-idf to convert our string-formatted data into a score determining the relevance of a word to a document in a collection of documents.

#### 3. Deep Encoding of Sparse Data
	- A simple neural network can be useful because of their ability to efficiently learn new ways to represent high-dimensional data/ sparse data.

#### 4. Determine the number of optimal clusters.
	- Clustering is the task of organising an unclassified collection of documents into meaningful, homogeneous clusters based on some concept of document similarity. 
<p align="center">
<img src="https://github.com/raofida75/Covid19-Literature-Clustering/blob/master/images/Silhoutte%20score.png" width="750"/>
</p>

#### 6. Apply t-SNE to Visualize the clusters.
	- To visualise the data, I needed to limit the number of columns to no more than three. To accomplish this, I have used t-SNE.


## Results 

[Link to the streamlit dashboard](https://share.streamlit.io/raofida75/covid19-literature-clustering/dashboard.py)


## Summary
In this project, I attempted to group similar Covid-19-related publications and visualise them after reducing the dimensions with the dimensionality reduction algorithm tSNE. Using streamlit, I was able to create an interactive scatterplot with similar publications grouped together. This method of grouping allows the health professional to stay up to date on new information related to the virus. The Kmeans algorithm was used to cluster the publications' body text after it had been preprocessed, vectorized, and deep encoded. Following that, topic modelling was applied to each cluster in order to identify relevant keywords in all of them. As a result, we obtained the topics that were prevalent in each cluster.
