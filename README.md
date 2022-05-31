# Covid19-Literature-Clustering

## Motivation
Given the rapid spread of Covid-19 and the abundance of research available on the internet, doctors may find it difficult to identify the relevant research. As a result, the goal of this project is to group similar articles together to make it easier to find similar publications. 

# Methodology -- [Data Preparation](https://github.com/raofida75/Covid19-Literature-Clustering/blob/master/Data%20Preparation.ipynb)
- Load metadata
- Get the path of all json files.
- Extract the body text of all the papers from the json files.
- Combine the metadata and the body text.
- Determine the language of each paper and keep only those in English.
- Remove null and duplicate values
