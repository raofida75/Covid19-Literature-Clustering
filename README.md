# Covid19-Literature-Clustering
[Link to the interactive dashboard](https://share.streamlit.io/raofida75/covid19-literature-clustering/dashboard.py)

With the increase in scientific literature, there was a need for those documents to be organised. Using machine learning techniques, I will organise and visualise the scientific literature on or related to COVID-19, so that papers on similar topics are grouped together. This simplifies the navigation of topics and related papers. I plan to use the well-known CORD-19 dataset to carry out this approach.

## [Data](https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge)
In response to the COVID-19 pandemic, the White House and a coalition of leading research groups have prepared the COVID-19 Open Research Dataset (CORD-19). CORD-19 is a resource of over 1,000,000 scholarly articles, including over 400,000 with full text, about COVID-19, SARS-CoV-2, and related coronaviruses. This freely available dataset is provided to the global research community to apply recent advances in natural language processing and other AI techniques to generate new insights in support of the ongoing fight against this infectious disease. There is a growing urgency for these approaches because of the rapid acceleration in new coronavirus literature, making it difficult for the medical research community to keep up.

## Methodology 
### [Data Preparation](https://github.com/raofida75/Covid19-Literature-Clustering/blob/master/Data%20Preparation.ipynb)
Following steps have been performed to prepare the data to perform clustering.
- Load metadata
- Get the path of all json files.
- Extract the body text of all the papers from the json files.
- Combine the metadata and the body text.
- Determine the language of each paper and keep only those in English.
- Remove null and duplicate values

## [Clustering Approach](https://github.com/raofida75/Covid19-Literature-Clustering/blob/master/Covid19%20Literature%20Clustering.ipynb)
### 1. Preprocess body text
- Remove Stop words
- Reduce inflected words to their base or root words in a process called Lematization.  It ensures that the root word (also known as a lemma) belongs to the language. The POS tag is used in the lemmatization process to determine the correct lemma.

### 2. Vectorization
After the data was pre-processed, it was time to convert it into a format that our algorithms can comprehend. To accomplish this, I have used tf-idf. This will convert our string-formatted data into a score determining the relevance of a word to a document in a collection of documents.

### 3. Deep Encoding Sparse Data
A quality clustering result is typically dependent on how well the data is organized. A simple neural network can be useful because of their ability to efficiently learn new ways to represent high-dimensional data/ sparse data.
I have built an autoencoder which will recreate the input it receives by first generating lower-dimensional data representation, which will preserve important features of the data. Lastly, I have reconstructed the data back to its original dimensions. This method can help us improve the clustering model's performance and reduce noise in the data.

### 4. Determine the number of optimal clusters.
Clustering is the task of organising an unclassified collection of documents into meaningful, homogeneous clusters based on some concept of document similarity. I have looked at the inertia and silhouette score at various k values to identify the ideal k value for k-means. When I plotted silhouette against k, there was be a k value where I got the highest score. This is the ideal number of clusters.
<p align="center">
<img src="https://github.com/raofida75/Covid19-Literature-Clustering/blob/master/Silhoutte%20score.png" width="750"/>
</p>

#### Result -- The optimal number of clusters found were 14. 

### 5. Evaluate different clustering algorithms using supervised learning techniques.
Now that we've determined an appropriate k value, we can then apply various clustering algorithms to the deep-encoded feature vector. Then, using supervised learning, I evaluated how well the clustering generalises. If the clustering algorithm was successful in identifying meaningful clusters in the data, training a classifier to predict which cluster a given paper belongs to should be possible. I used the Stochastic Gradient Descent classifier to assess the generalizability of the clustering algorithm.
Clustering algorithms evaluated includes:
1. Agglomerative Clustering
2. K Means Clustering
3. Spectral Clustering

#### Results 
Evaluation of Kmeans clustering algorithm on SGD Classifier: 

	Accuracy score: 87.0 % 
	F1 score: 87.0 %
	Precision: 87.0 % 
	Recall: 87.0 %


Evaluation of SpectralClustering clustering algorithm on SGD Classifier: 

	Accuracy score: 87.0 % 
	F1 score: 87.0 %
	Precision: 87.0 % 
	Recall: 87.0 %


Evaluation of AgglomerativeClustering clustering algorithm on SGD Classifier: 

	Accuracy score: 78.0 % 
	F1 score: 78.0 %
	Precision: 79.0 % 
	Recall: 78.0 %
On the test dataset, both Kmeans and Spectral clustering were able to generalise well. However, when I looked at the data distribution across different clusters in spectral clustering, I noticed that most of the data was concentrated in a few clusters, whereas Kmeans spreads the data out well. As a result, I proceeded with Kmeans.

### 6. Visualize the clusters.
After assigning a cluster to each data point, the next logical step was to visualise it. But there's a catch. We have 100 columns of data. To visualise it, I needed to limit the number of columns to no more than three. To accomplish this, I have used t-SNE.

<p align="center">
<img src="https://github.com/raofida75/Covid19-Literature-Clustering/blob/master/Clusters%20TSNE.png" width="1000"/>
</p>

### 7. Topic modelling on each cluster
Lastly, I have tried to determine which words in each cluster are the most important. The articles were clustered using K-means, but the themes were not labelled. I have now performed topic modelling to determine the most important terms for each cluster.

I have used Non-Negative Matrix Factorization (NMF) for topic modelling.


## Summary
In this project, I attempted to group similar Covid-19-related publications and visualise them after reducing the dimensions with the dimensionality reduction algorithm tSNE. Using streamlit, I was able to create an interactive scatterplot with similar publications grouped together. This method of grouping allows the health professional to stay up to date on new information related to the virus. The Kmeans algorithm was used to cluster the publications' body text after it had been preprocessed, vectorized, and deep encoded. Following that, topic modelling was applied to each cluster in order to identify relevant keywords in all of them. As a result, we obtained the topics that were prevalent in each cluster.
