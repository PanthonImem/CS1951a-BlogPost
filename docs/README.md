# CS1951A Final Project

## Introduction
Wikipedia is a rich source of human-readable information on a vast variety of subjects. Automatic extraction of its content should prove useful for many data related projects. We aim to extract insights from Wikipedia articles on people (David Bowie, etc.) using natural language processing (NLP) techniques. 

We hope, at least, to be able to classify a person into distinct categories (extracted, perhaps, using tf-idf) based on the content of his/her Wikipedia article. For example, the topics extracted from the database using Latent Dirichlet Allocation should give insights on ways to classify a person. We want to be able to tell, for example, based on the topic distributions if the person described in a Wikipedia article is a politician, an artist, or a singer, etc.
 
As probability vectors, LDA distributions also give us embeddings into high-dimensional space from which we can find embeddings into 2 or 3 dimensions for visualization. Then, if time permits, we may also learn to generate fake Wikipedia articles using Markov models and GANs.

## Blog Post 1

Every person was born on some day. The important ones have Wikipedia articles. And the really important ones have Wikipedia articles that list their birthdays. (https://en.wikipedia.org/wiki/Category:20th-century_births)[lol]


The most insightful part of a Wikipedia article is the first section, the so called introduction. In our case, the first section tells us what the person does, what they are known for and, in general, why they deserve a Wikipedia article. 


 
### Keyword Extraction: Term Frequency-Inverse Document Frequency(TF-IDF)

From all first paragraphs of these 10,000 pages, we have 5461 unique words in the extracted data. Among these 3815 are common english words(not name of person or place, result obtained by comparing words extracted to nltk library of common english word).
 
We want to know which word is important so we will use a technique called TF-IDF. 

The idea is, we believe that words that are important will appear more than others(Term Frequency), while it should
not be so common that it appears in too many documents(Inverse Document Frequency). 

The simplest attempt following the standard TF-IDF formular(Term Frequency * log(totaldoc/docfreq)) yielded a non-particularly interesting result:

<p align="center">
 <img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/wordcloud_bad_names.png" >
</p>

We can see the the most important words are names. Thus we would want to clean out any word not in common english word. 
Here is the result:

<p align="center">
 <img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/wordcloud_bad_words.png" >
</p>

Oh No! We can see that we are getting weird words that we are not expecting. 

Upon closer inspection, these words only appear in very few documents so their IDFs is very high despite the fact that they do not appear a lot. This is not what we want, thus we will experiment with a couple of weighting methods. 

<p align="center">
 <img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/wordcloud_alternate1.png" >
</p>

After a few different experiment with weighting, we ended up with a method to calculate idf which wikipedia calls Probabilistic IDF (\*include equation).

<p align="center">
 <img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/wordcloud.png" >
</p>

### Similarity Search: Nearest Neighbor Search with 'Categories' Properties

From 10,000 most viewed Wikipedia pages in People category, 9,620 pages has 'Categories' property that can be accessed by requesting to Wikipedia API. 

To illustrate, Wikipedia has 'Michael Jackson' fit the following categories: 
'1958 births', '2009 deaths', '20th-century American singers', '21st-century American singers', 'AC with 15 elements', 'Accidental deaths in California', 'African-American choreographers', 'African-American dancers', 'African-American male dancers', 'African-American male singers'

Since 'Categories' property summarizes each person and focus on their occupations, nationalities, and ethnicities, we hope that we can classify people into groups using their categories already made by Wikipedia.

Firstly, we tried Nearest Neighbor Search to find the closest match of each person. In the data preparation process, we discard categories that do not relate to the person's identity. Those categories commonly contains the words 'articles', 'pages', etc. Then, we use a one-hot encoding i.e. dummy variable to convert categories into integer data. After the preparation, we have sklearn.NearestNeighbor fit the one-hot data and predict top two closest match of each person. 

In the first try, we have a dummy variable for every category that appears more than 7 times. The predictions of the 17 most viewed Wikipedia pages are as follow. 

<p align="center">
 <img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/untokenized.jpg">
</p>

To improve the predictions, we tokenize the categories and have important words representing each person instead. This is because we can create subcategories from the categories given by Wikipedia. For instance, with tokenized version, Michal Jackson will belong to 'American', 'Africa-American', 'dancers', 'singers', etc. Without the tokenization, Michael Jackson would never be matched to an American dancer even though they are similar. 

Thus, we tokenize importanct categories and perform Nearest Neighbor Search with the new one-hot matrix. The predictions from this try is more accurate.

<p align="center">
 <img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/tokenized.jpg">
</p>

We also plotted this one-hot of tokenized categories hoping for distinct clusterings. The graph is shown below. 

<p align="center">
 <img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/tokenized_plot.jpg" width="450">
</p>

We might be able to improve Nearest Neighbor Search by adding weights to similar words. Apart from NNS, we plan to use the categories property provided by Wikipedia to classify people into groups. Possibly, we can present common features of pages that have high views by exploring the categories.


### Document Clustering: K-means Clustering with Doc2vec

We also experimented with document clustering. We first turn Wikipedia pages to numeric representations using doc2vec. doc2vec is a modified word2vec algorithm, designed to capture semantic information of a document. As an addition to word2vec, a document ID is concatenated to the input word sequence. The learned embedding of each document ID is the vector representation of the document.

Once we obtain the representations of all the documents, we utilize the k-means algorithm to cluster the documents. We ran doc2vec and k-means on the dataset of 10,000 Wikipedia pages. Setting the number of clusters to 10, we obtained the following results with doc2vec.

<p align="center">
	<table style="width: 100%; border: 1px solid black; border-collapse: collapse;">
    	<tr style="border: 1px solid black">
          <th>Cluster</th>
          <th width="40%">Word Cloud</th> 
          <th>Example</th>
          <th>Number of Pages</th>
          <th>Variance</th>
        </tr>
        <tr style="border: 1px solid black">
          <td>1</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/clusters_n10_doc2vec/1.jpg" width="500"></td> 
          <td>LeBron James, Dwayne Johnson, Tonya Harding, Khabib Nurmagomedov, John Cena, Stephen Curry, Anthony Joshua, Caitlyn Jenner, Naomi Osaka, The Undertaker</td>
          <td>814</td>
          <td>0.0188</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>2</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/clusters_n10_doc2vec/2.jpg" width="500"></td> 
          <td>Anthony Bourdain, Meghan, Duchess of Sussex, Charles, Prince of Wales, Prince Philip, Duke of Edinburgh, Pablo Escobar, Jeff Bezos, Diana, Princess of Wales, Prince William, Duke of Cambridge, Jeffrey Dahmer, Prince Harry, Duke of Sussex</td>
          <td>956</td>
          <td>0.0188</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>3</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/clusters_n10_doc2vec/3.jpg" width="500"></td> 
          <td>Louis Tomlinson, Antonio Maria Magro, Dua Lipa, Pete Davidson, Andrew Cunanan, Hailey Baldwin, Dolores O'Riordan, John Paul Getty III, Lisa Bonet, Beto O'Rourke</td>
          <td>3597</td>
          <td>0.0055</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>4</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/clusters_n10_doc2vec/4.jpg" width="500"></td> 
          <td>Elon Musk, Stephen Hawking, P. T. Barnum, Albert Einstein, Ted Kaczynski, Steve Jobs, William Shakespeare, Bill Gates, Rajneesh, Nikola Tesla</td>
          <td>342</td>
          <td>0.0322</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>5</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/clusters_n10_doc2vec/5.jpg"></td> 
          <td>Donald Trump, George H. W. Bush, John McCain, Winston Churchill, Barack Obama, Adolf Hitler, Brett Kavanaugh, Mahatma Gandhi, George W. Bush, Martin Luther King Jr.</td>
          <td>336</td>
          <td>0.0429</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>6</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/clusters_n10_doc2vec/6.jpg"></td> 
          <td>Elizabeth II, Queen Victoria, Princess Margaret, Countess of Snowdon, George VI, Mary, Queen of Scots, Edward VIII, George V, Elizabeth I of England, Joaquín "El Chapo" Guzmán, Henry VIII of England</td>
          <td>253</td>
          <td>0.0428</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>7</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/clusters_n10_doc2vec/7.jpg"></td> 
          <td>Cristiano Ronaldo, Lionel Messi, Michael Jordan, Tom Brady, Kylian Mbappé, Mohamed Salah, Conor McGregor, Roger Federer, Virat Kohli, Neymar</td>
          <td>387</td>
          <td>0.0414</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>8</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/clusters_n10_doc2vec/8.jpg"></td> 
          <td>Cardi B, Freddie Mercury, XXXTentacion, Ariana Grande, 6ix9ine, Avicii, Donald Glover, Nick Jonas, Post Malone, Michael Jackson</td>
          <td>697</td>
          <td>0.0269</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>9</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/clusters_n10_doc2vec/9.jpg"></td> 
          <td>Jason Momoa, Stan Lee, Sylvester Stallone, Jennifer Aniston, Michael B. Jordan, Burt Reynolds, Ryan Reynolds, Chris Hemsworth, Josh Brolin, Gianni Versace</td>
          <td>1930</td>
          <td>0.0106</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>10</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/clusters_n10_doc2vec/10.jpg"></td> 
          <td>Priyanka Chopra, Sridevi, Tom Cruise, Demi Lovato, Clint Eastwood, Scarlett Johansson, Emily Blunt, Keanu Reeves, Bradley Cooper, John Krasinski</td>
          <td>688</td>
          <td>0.0247</td>
        </tr>
    </table>
</p>

Clustering allows us to better understand out dataset. From the table, we see that there are 6 main "categories" in our dataset: (1) sports (Clusters 1, 7), (2) royalty (Clusters 2, 6), (3) actors/actesses, movies-related figures (Clusters 3, 9, 10), (4) successful figures (Cluster 4), (5) politic figures (Cluster 5), and (6) singers (Cluster 8). From the word clouds, we also see that the important words are corresponding to each cluster. For instance, the word "work" was extracted as the most important word in Cluster 4, which represent those best known for their career successes such as Elon Musk, Albert Einstein and William Shakespeare. Moreover, we notice that the dataset is highly imbalanced as shown in the chart below. More than half of the dataset are actors/actresses or movies-related figures, while only 3% are political figures. 

<p align="center">
 <img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/clusters_n10_doc2vec/pie_.jpg">
</p>

 However, there are some issues concerning our clustering. The first issue is a trade off between a large and small numbers of clusters. When we have too many clusters, some categories might be duplicated or very similar as the results we obtained when having 10 clusters. On the other hand, when we have a small number of clusters, a cluster may include irrelevant pages. As an example, Anthony Bourdain who was a chef was put in the royal families cluster (Cluster 2). While solving the trade off problem is not trivial, a possible work around for this is using alternative embeddings which better represent information about each cluster. As an idea, if we want to cluster documents based on occupations, we can build a neural network which enbeds documents and is trained to predict occupations.





