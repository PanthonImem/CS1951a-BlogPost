# CS1951A Final Project

## Introduction
Wikipedia is a rich source of human-readable information on a vast variety of subjects. Automatic extraction of its content will prove useful for many data related projects. We aim to extract insights from Wikipedia articles on people (David Bowie, etc.) using natural language processing (NLP) techniques. 

We hope, at least, to be able to classify a person into distinct categories (extracted, perhaps, using tf-idf) based on the content of his/her Wikipedia article. For example, the topics extracted from the database using Latent Dirichlet Allocation should give insights on ways to classify a person. We want to be able to tell, for example, based on the topic distributions if the person described in a Wikipedia article is a politician, an artist, or a singer, etc.
 
As probability vectors, LDA distributions also give us embeddings into high-dimensional space from which we can find embeddings into 2 or 3 dimensions for visualization. Then, if time permits, we may also learn to generate fake Wikipedia articles using Markov models and GANs.

## Blog Post 1

The dataset we are interested is the 10,000 most viewed Wikipedia pages in People category.
 
 
 We have 5461 unique words in the extracted data. 
 Among these 3815 are common english words(not name of person or place, 
 result obtained by comparing words extracted to nltk library of common english word). 
 
### Important Word Extraction: Term Frequency-Inverse Document Frequency(TF-IDF)
We want to know which word is important so we will use a technique called TF-IDF. 

The idea is, we believe that words that are important will appear more than others(Term Frequency), while it should
not be so common that it appears in too many documents(Inverse Document Frequency). 

The simplest attempt following the standard TF-IDF formular(Term Frequency * log(totaldoc/docfreq)) yielded a non-particularly interesting result:

![alt text](https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/wordcloud_bad_names.png)

We can see the the most important words are names. Thus we would want to clean out any word not in common english word. 
Here is the result:

![alt text](https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/wordcloud_bad_words.png)

Oh No! We can see that we are getting weird words that we are not expecting. 

Upon closer inspection, these words only appear in very few documents so their IDFs is very high despite the fact that they do not appear a lot. This is not what we want, thus we will experiment with a couple of weighting methods. 

![alt text](https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/wordcloud_alternate1.png)

After a few different experiment with weighting, we ended up with a method to calculate idf which wikipedia calls Probabilistic IDF (*include equation).

![alt text](https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/wordcloud.png)

### Similarity Search: Nearest Neighbor Search with 'Categories' Properties

From 10,000 most viewed Wikipedia pages in People category, 9,620 pages has 'Categories' property that can be accessed by requesting to Wikipedia API. 

To illustrate, Wikipedia has 'Michael Jackson' fit the following categories: 
'1958 births', '2009 deaths', '20th-century American singers', '21st-century American singers', 'AC with 15 elements', 'Accidental deaths in California', 'African-American choreographers', 'African-American dancers', 'African-American male dancers', 'African-American male singers'

Since 'Categories' property summarizes each person and focus on their occupations, nationalities, and ethnicities, we hope that we can classify people into groups using their categories already made by Wikipedia.

Firstly, we tried Nearest Neighbor Search to find the closest match of each person. In the data preparation process, we discard categories that do not relate to the person's identity. Those categories commonly contains the words 'articles', 'pages', etc. Then, we use a one-hot encoding i.e. dummy variable to convert categories into integer data. After the preparation, we have sklearn.NearestNeighbor fit the one-hot data and predict top two closest match of each person. 

In the first try, we have a dummy variable for every category that appears more than 7 times. The predictions of the 17 most viewed Wikipedia pages are as follow. 

![alt text](https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/untokenized.jpg)

To improve the predictions, we tokenize the categories and have important words representing each person instead. This is because we can create subcategories from the categories given by Wikipedia. For instance, with tokenized version, Michal Jackson will belong to 'American', 'Africa-American', 'dancers', 'singers', etc. Without the tokenization, Michael Jackson would never be matched to an American dancer even though they are similar. 

Thus, we tokenize importanct categories and perform Nearest Neighbor Search with the new one-hot matrix. The predictions from this try is more accurate.

![alt text](https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/tokenized.jpg)

We also try plotting the one-hot of tokenized categories of pages to see if there are distinct clusterings, but the graph shows that there is not. 


![alt text](https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/tokenized_plot.jpg | width=100)

      
