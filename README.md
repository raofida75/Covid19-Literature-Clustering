# Covid19-Literature-Clustering

With the increase in scientific literature, there was a need for those documents to be organised. Using machine learning techniques, I will organise and visualise the scientific literature on or related to COVID-19, so that papers on similar topics are grouped together. This simplifies the navigation of topics and related papers. I plan to use the well-known CORD-19 dataset to carry out this approach.

## Methodology 
### [Data Preparation](https://github.com/raofida75/Covid19-Literature-Clustering/blob/master/Data%20Preparation.ipynb)
- Load metadata
- Get the path of all json files.
- Extract the body text of all the papers from the json files.
- Combine the metadata and the body text.
- Determine the language of each paper and keep only those in English.
- Remove null and duplicate values

## Clustering Approach
### 1. Preprocess body text
- Remove Stop words
- Reduce inflected words to their base or root words in a process called Lematization.  It ensures that the root word (also known as a lemma) belongs to the language. The POS tag is used in the lemmatization process to determine the correct lemma.

### 2. Vectorization
After the data has been pre-processed, it is time to convert it into a format that our algorithms can comprehend. To accomplish this, we'll use tf-idf. This will convert our string-formatted data into a score determining the relevance of a word to a document in a collection of documents.

### 3. Deep Encoding Sparse Data
A quality clustering result is typically dependent on how well the data is organized. A simple neural network can be useful because of their ability to efficiently learn new ways to represent high-dimensional data/ sparse data.
I will be building an autoencoder which will recreate the input it receives by first generating lower-dimensional data representation, which will preserve important features of the data. Lastly, we will reconstruct the data back to its original dimensions. This method can help us improve the clustering model's performance and reduce noise in the data.

### 4. Determine the number of optimal clusters.
Clustering is the task of organising an unclassified collection of documents into meaningful, homogeneous clusters based on some concept of document similarity. We'll look at the inertia and silhouette score at various k values to identify the ideal k value for k-means.When we plot silhouette against k, there will be a k value where we get the highest score. This is the ideal number of clusters.

#### Result -- The optimal number of clusters found were 14. 

### 5. Evaluate different clustering algorithms using supervised learning techniques.
Now that we've determined an appropriate k value, we can apply various clustering algorithms to the deep-encoded feature vector. Then, using supervised learning, we'll see how well the clustering generalises. If the clustering algorithm was successful in identifying meaningful clusters in the data, it should be possible to train a classifier to predict which cluster a given paper belongs to. To evaluate the generalizability of the clustering algorithm, I will use the Stochastic Gradient Descent classifier.
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
On the test dataset, both Kmeans and Spectral clustering appear to generalise well. However, when I looked at the data distribution across different clusters in spectral clustering, I noticed that most of the data was concentrated in a few clusters, whereas Kmeans spreads the data out well. As a result, I'll go with Kmeans.

### 6. Visualize the clusters.
After assigning a cluster to each data point, the next logical step is to visualise it. But there's a catch. We have 100 columns of data. To visualise it, we'll need to limit the number of columns to no more than three. To accomplish this, we'll use t-SNE.
<img src="https://github.com/raofida75/Covid19-Literature-Clustering/blob/master/Clusters%20TSNE.png" width="1000"/>
