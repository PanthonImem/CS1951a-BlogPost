{% include mathjax.html %}

# CS1951A Final Project

[blog-1](blog-1.md)
[blog-2](blog-2.md)
[blog-3](blog-3.md)

## Introduction
Wikipedia is a rich source of human-readable information on a vast variety of subjects. Automatic extraction of its content should prove useful for many data related projects. We aim to extract insights from Wikipedia articles on people (David Bowie, etc.) using natural language processing (NLP) techniques. 

We hope, at least, to be able to classify a person into distinct categories (extracted, perhaps, using tf-idf) based on the content of his/her Wikipedia article. For example, the topics extracted from the database using Latent Dirichlet Allocation should give insights on ways to classify a person. We want to be able to tell, for example, based on the topic distributions if the person described in a Wikipedia article is a politician, an artist, or a singer, etc.

As probability vectors, LDA distributions also give us embeddings into high-dimensional space from which we can find embeddings into 2 or 3 dimensions for visualization. Then, if time permits, we may also learn to generate fake Wikipedia articles using Markov models and GANs.

## Blog Post 1 

Wikipedia put articles about people with listed birthdates into a list [here](https://en.wikipedia.org/wiki/Category:Births_by_century). Scraping pages off it gives us a huge collection of links, 1.37 million in total, to Wikipedia articles about people. One million is hardly a big number when it comes to data science. But it is a big number for the collective memory. Humanity really only remembers the big blots in its history like Shakespeare or Hitler or Napoléon, and stars that just went out like Stephen Hawking. There are not a million people like those. To sift away the forgotten, we ask Wikipedia for page views which can be done by making a million requests like [this one](https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/all-agents/Fred_Rogers/monthly/20180101/20181231) we made for Mister Rogers. These numbers tell us where the limelight shines. 

| Rank | Name              | Views    |
| ---: | :---------------- | :------- |
|    1 | Louis Tomlinson   | 32647670 |
|    2 | Freddie Mercury   | 22261530 |
|    3 | Elizabeth II      | 20049022 |
|    4 | Stephen Hawking   | 19025060 |
|    5 | Donald Trump      | 18663083 |
|    6 | Cristiano Ronaldo | 18467102 |
|    7 | Cardi B           | 17955598 |
|    8 | Elon Musk         | 17836456 |
|    9 | XXXTentacion      | 15249774 |
|   10 | Lionel Messi      | 13457818 |
|   11 | LeBron James      | 12555210 |
|   12 | Ariana Grande     | 12307383 |
|   13 | Jason Momoa       | 12208869 |
|   14 | 6ix9ine           | 12091008 |
|   15 | George H.W. Bush  | 12078037 |
|   16 | Anthony Bourdain  | 11867794 |
|   17 | Priyanka Chopra   | 11613801 |
|   18 | John McCain       | 11550433 |
|   19 | Queen Victoria    | 11415901 |
|   20 | Stan Lee          | 11352622 |

For those who asked the same question we did (with or without expletives), [Louis Tomlinson](https://en.wikipedia.org/wiki/Louis_Tomlinson) is a member of _One Direction_. How he came to have 15 times more page views than his other band members remains a mystery beyond the reach of our intelligence. Taking data as fact, sorting the page views in decreasing order gives us the graph below showing a rapidly decaying number of views as you fall down the popularity ladder.


 <img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/page-views.png" style="width:95%;">

We took the $$n=10000$$ most popular articles as our subset, which corresponds to more or less about half of all page views in 2018. (Here, the top 1% do own half the wealth.) The cutoff at $$10000$$ is really quite arbitrary. But we want a small subset for two main reasons:

  - Articles with more views are better maintained, and
  - Most people don't care about the rest.

In addition to graciously providing views statistics, Wikipedia's API endpoint also extracts plaintext from Wikipedia markup language when asked [nicely](https://en.wikipedia.org/w/api.php?action=query&prop=categories%7Cextracts&titles=Srinivasa+Ramanujan&explaintext=true&format=json&cllimit=5000). At this point, we know who we want in our dataset, the link to their page, and how to extract plaintext from it, collecting data is simply a matter of waiting for page requests.

Now, it has never been a matter of debate that the most insightful part of a Wikipedia article is the so-called [lead](https://en.wikipedia.org/wiki/Wikipedia:Manual_of_Style/Lead_section). In our case, the lead section tells us what the person does, what they are known for and, in general, why they deserve a Wikipedia article. So for most of the projects in this blog post, we use the stopped-tokenized-lemmatized bag-of-words representation of the lead section.

### Keyword Extraction: Term Frequency-Inverse Document Frequency

Altogether, the $$10000$$ lead sections contain 5461 unique words. To quantify the importance of each word, we use a popular keyword extraction technique called term [frequency-inverse document frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) (tf-idf.) 

The main insight of tf-idf is that important words should give information about the articles containing them. In particular, they should appear rather frequently in documents that contain them, but not so ubiquitously that they show up in every other article. The simplest of the tf-idf formulae articulating this train of thought is

$$ \text{tf-idf}(w) = 
  \text{frequency of } w
  \cdot 
  \log \left(
    \dfrac{N}{\text{frequency of documents containing } w}
  \right)
$$

Using the tf-idf formula above to rank the words gives us an expected result, which we show in the wordcloud below.

<p align="center">
 <img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/wordcloud_bad_names.png" >
</p>

We can see the the most important words are names, which makes sense because important words should distinguish between articles. But not very interesting because we already know that these words are important. To counteract this phenomenon, we only consider common English words, i.e., words you would see in an English dictionary. This leaves us with 3815 unique words, the wordcloud of whose is shown below.

<p align="center">
 <img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/wordcloud_bad_words.png" >
</p>

Words like "cave" and "pompey" and "vaccine" definitely tell us a lot about the articles containing them, but they don't cover enough of the corpus to give us a big picture. (If you read on, you will see that over half of the articles are about actors or actresses, but it's impossible to tell just looking at these words that there are any famous actors.) Upon closer inspection, these words only appear in very few documents, so their idf-weights are very high despite not appearing a lot. We experimented with other tf-idf formulae:

<p align="center">
 <img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/wordcloud_alternate1.png" >
</p>

After a few different experiment with weighting, we ended up with a method to calculate idf which wikipedia calls probabilistic idf.

  $$ idf(w) = \log {\frac {N-n_{w}}{n_{w}}} $$

where $$n_{w}$$ is the number of articles containing the term $$w$$. 



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
  <table style="width: 120%; border: 1px solid black; border-collapse: collapse; margin-left: -65px;">
      <tr style="border: 1px solid black">
          <th>Cluster</th>
          <th width="40%">Word Cloud</th> 
          <th>Example</th>
          <th>Number of Pages</th>
          <th>Variance</th>
        </tr>
        <tr style="border: 1px solid black">
          <td>1</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/clusters_n10_doc2vec/1.jpg" style="width: 250px; vertical-align: top"></td> 
          <td>LeBron James, Dwayne Johnson, Tonya Harding, Khabib Nurmagomedov, John Cena, Stephen Curry, Anthony Joshua, Caitlyn Jenner, Naomi Osaka, The Undertaker</td>
          <td>814</td>
          <td>0.0188</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>2</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/clusters_n10_doc2vec/2.jpg" style="width: 250px; vertical-align: top"></td> 
          <td>Anthony Bourdain, Meghan, Duchess of Sussex, Charles, Prince of Wales, Prince Philip, Duke of Edinburgh, Pablo Escobar, Jeff Bezos, Diana, Princess of Wales, Prince William, Duke of Cambridge, Jeffrey Dahmer, Prince Harry, Duke of Sussex</td>
          <td>956</td>
          <td>0.0188</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>3</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/clusters_n10_doc2vec/3.jpg" style="width: 250px; vertical-align: top"></td> 
          <td>Louis Tomlinson, Antonio Maria Magro, Dua Lipa, Pete Davidson, Andrew Cunanan, Hailey Baldwin, Dolores O'Riordan, John Paul Getty III, Lisa Bonet, Beto O'Rourke</td>
          <td>3597</td>
          <td>0.0055</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>4</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/clusters_n10_doc2vec/4.jpg" style="width: 250px; vertical-align: top"></td> 
          <td>Elon Musk, Stephen Hawking, P. T. Barnum, Albert Einstein, Ted Kaczynski, Steve Jobs, William Shakespeare, Bill Gates, Rajneesh, Nikola Tesla</td>
          <td>342</td>
          <td>0.0322</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>5</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/clusters_n10_doc2vec/5.jpg" style="width: 250px; vertical-align: top"></td> 
          <td>Donald Trump, George H. W. Bush, John McCain, Winston Churchill, Barack Obama, Adolf Hitler, Brett Kavanaugh, Mahatma Gandhi, George W. Bush, Martin Luther King Jr.</td>
          <td>336</td>
          <td>0.0429</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>6</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/clusters_n10_doc2vec/6.jpg" style="width: 250px; vertical-align: top"></td> 
          <td>Elizabeth II, Queen Victoria, Princess Margaret, Countess of Snowdon, George VI, Mary, Queen of Scots, Edward VIII, George V, Elizabeth I of England, Joaquín "El Chapo" Guzmán, Henry VIII of England</td>
          <td>253</td>
          <td>0.0428</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>7</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/clusters_n10_doc2vec/7.jpg" style="width: 250px; vertical-align: top"></td> 
          <td>Cristiano Ronaldo, Lionel Messi, Michael Jordan, Tom Brady, Kylian Mbappé, Mohamed Salah, Conor McGregor, Roger Federer, Virat Kohli, Neymar</td>
          <td>387</td>
          <td>0.0414</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>8</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/clusters_n10_doc2vec/8.jpg" style="width: 250px; vertical-align: top"></td> 
          <td>Cardi B, Freddie Mercury, XXXTentacion, Ariana Grande, 6ix9ine, Avicii, Donald Glover, Nick Jonas, Post Malone, Michael Jackson</td>
          <td>697</td>
          <td>0.0269</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>9</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/clusters_n10_doc2vec/9.jpg" style="width: 250px; vertical-align: top"></td> 
          <td>Jason Momoa, Stan Lee, Sylvester Stallone, Jennifer Aniston, Michael B. Jordan, Burt Reynolds, Ryan Reynolds, Chris Hemsworth, Josh Brolin, Gianni Versace</td>
          <td>1930</td>
          <td>0.0106</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>10</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/clusters_n10_doc2vec/10.jpg" style="width: 250px; vertical-align: top"></td> 
          <td>Priyanka Chopra, Sridevi, Tom Cruise, Demi Lovato, Clint Eastwood, Scarlett Johansson, Emily Blunt, Keanu Reeves, Bradley Cooper, John Krasinski</td>
          <td>688</td>
          <td>0.0247</td>
        </tr>
    </table>
</p>

Clustering allows us to better understand out dataset. From the table, we see that there are 6 main "categories" in our dataset: (1) sports (Clusters 1, 7), (2) royalty (Clusters 2, 6), (3) actors/actesses, movies-related figures (Clusters 3, 9, 10), (4) successful figures (Cluster 4), (5) politic figures (Cluster 5), and (6) singers (Cluster 8). From the wordclouds, we also see that the important words are corresponding to each cluster. For instance, the word "work" was extracted as the most important word in Cluster 4, which represent those best known for their career successes such as Elon Musk, Albert Einstein and William Shakespeare. Moreover, we notice that the dataset is highly imbalanced as shown in the chart below. More than half of the dataset are actors/actresses or movies-related figures, while only 3% are political figures. 

<p align="center">
 <img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/clusters_n10_doc2vec/pie_.jpg">
</p>

 However, there are some issues concerning our clustering. The first issue is a trade off between a large and small numbers of clusters. When we have too many clusters, some categories might be duplicated or very similar as the results we obtained when having 10 clusters. On the other hand, when we have a small number of clusters, a cluster may include irrelevant pages. As an example, Anthony Bourdain who was a chef was put in the royal families cluster (Cluster 2). While solving the trade off problem is not trivial, a possible work around for this is using alternative embeddings which better represent information about each cluster. As an idea, if we want to cluster documents based on occupations, we can build a neural network which enbeds documents and is trained to predict occupations.

## Blog Post 2
### Document Clustering: Spectral Clustering with Bag of Words 

To improve the performance of document clustering, we tried clustering socuments with [spectral clustering](https://en.wikipedia.org/wiki/Spectral_clustering). Instead of grouping points that are close to one another together like K-means approach, spectral clustering clusters points that connect to one another together. With many feature extraction methods including doc2vec and TF-IDF, spectral clustering with bag of words gives the best result. Unlike the result from Kmeans with doc2vec, this result groups athletes who play different sports. However, there is no group of royalty in this clustering result. 

The result of spectral clustering with bag of words is the following. 


<p align="center">
  <table style="width: 120%; border: 1px solid black; border-collapse: collapse; margin-left: -65px;">
      <tr style="border: 1px solid black">
          <th>Cluster</th>
          <th width="40%">Word Cloud</th> 
          <th>Example</th>
          <th>Number of Pages</th>
          <th>Variance</th>
        </tr>
        <tr style="border: 1px solid black">
          <td>1</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/spectralcluster_n10_wordcloud/1.png" style="width: 250px; vertical-align: top"></td> 
          <td>Elizabeth II, Freddie Mercury, Elon Musk, Stephen Hawking, Anthony Bourdain, Queen Victoria, Stan Lee, Princess Margaret, Countess of Snowdon,  Charles (Prince of Wales), Prince Philip (Duke of Edinburgh)</td>
          <td>987</td>
          <td>3.09e-09</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>2</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/spectralcluster_n10_wordcloud/2.png" style="width: 250px; vertical-align: top"></td> 
          <td>Antonio Maria Magro, Pete Davidson, Andrew Cunanan, Gianni Versace, Hailey Baldwin, Jeffrey Dahmer, Dolores O'Riordan, John Paul Getty III, Lisa Bonet, Chadwick Boseman</td>
          <td>3474</td>
          <td>4.60e-10</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>3</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/spectralcluster_n10_wordcloud/3.png" style="width: 250px; vertical-align: top"></td> 
          <td>Dwayne Johnson, Tonya Harding, Jeff Bezos, Khabib Nurmagomedov, Conor McGregor, Roger Federer, John Cena, Anthony Joshua, Serena Williams, Muhammad Ali</td>
          <td>377</td>
          <td>2.60e-09</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>4</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/spectralcluster_n10_wordcloud/4.png" style="width: 250px; vertical-align: top"></td> 
          <td>Donald Trump, George H. W. Bush, John McCain, Winston Churchill, Barack Obama, P. T. Barnum, Brett Kavanaugh, George W. Bush, John F. Kennedy, Abraham Lincoln</td>
          <td>602</td>
          <td>3.30e-09</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>5</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/spectralcluster_n10_wordcloud/5.png" style="width: 250px; vertical-align: top"></td> 
          <td>Tom Cruise, Demi Lovato, Scarlett Johansson, Emily Blunt, Bradley Cooper, Leonardo DiCaprio, Rami Malek, Ellen DeGeneres, Elizabeth Olsen, Robin Williams</td>
          <td>715</td>
          <td>1.55e-09</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>6</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/spectralcluster_n10_wordcloud/6.png" style="width: 250px; vertical-align: top"></td> 
          <td>LeBron James, Michael Jordan, Tom Brady, Stephen Curry, Shaquille O'Neal, Kobe Bryant, Nick Foles, O. J. Simpson, Kevin Durant, Patrick Mahomes</td>
          <td>507</td>
          <td>1.78e-09</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>7</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/spectralcluster_n10_wordcloud/7.png" style="width: 250px; vertical-align: top"></td> 
          <td>Cristiano Ronaldo, Lionel Messi, Kylian Mbappé, Mohamed Salah, Virat Kohli, Neymar, Harry Kane, Zlatan Ibrahimović, David Beckham, Pelé</td>
          <td>470</td>
          <td>2.15e-09</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>8</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/spectralcluster_n10_wordcloud/8.png" style="width: 250px; vertical-align: top"></td> 
          <td>Cardi B, Louis Tomlinson, XXXTentacion, Ariana Grande, 6ix9ine, Avicii, Donald Glover, Nick Jonas, Post Malone, Michael Jackson</td>
          <td>727</td>
          <td>2.32e-09</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>9</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/spectralcluster_n10_wordcloud/9.png" style="width: 250px; vertical-align: top"></td> 
          <td>Meghan, Duchess of Sussex, Jennifer Aniston, Kylie Jenner, Michael B. Jordan, Burt Reynolds, John Krasinski, Tom Hardy, Chris Pratt, Jennifer Lawrence, Fred Rogers</td>
          <td>1349</td>
          <td>1.07e-09</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>10</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/spectralcluster_n10_wordcloud/10.png" style="width: 250px; vertical-align: top"></td> 
          <td>Jason Momoa, Priyanka Chopra, Sridevi, Sylvester Stallone, Clint Eastwood, Ryan Reynolds, Chris Hemsworth, Josh Brolin, Keanu Reeves, Mila Kunis</td>
          <td>792</td>
          <td>1.62e-09</td>
        </tr>
    </table>
</p>

With this clustering, 55% of the dataset are actors/actresses, 22% are athletes, 6% are politicians, and 7% are singers. Royalties and successful people with other careers are in the rest 10%. 

<p align="center">
 <img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/spectral_clustering_piechart.png" width="300">
</p>

To roughly confirm the accuracy of the model, we compared the results with percentage of 10,000 most viewed Wikipedia articles that belong to each occupation using category property that Wikipedia gives to each its article, which presents in the pie chart below.

<p align="center">
 <img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/categories_piechart.png" width="300">
</p>

Compared two clusterings with the pie chart using keywords from categories attribute, we can see that the spectral clustering with bag of words is better at clustering actors/actresses and singers, while kmeans clustering with doc2vec is better at grouping athletes and politicians. 

The pie chart obtained from keywords in categories is however not completely accurate itself because, firstly, one person may have more than one occupation indicated in the category property; secondly, Wikipedia only give 9,620 pages from 10,000 most viewed Wikipedia articles in people the category property; and lastly, to compare with the results from k-mean algorithm, we disregarded several jobs, including novelists, comedians, and artists. 

To further develop the accuracy test, we should create labels for each person. One possible method is to detect a keyword (i.e. actor, actress, government, politician, singer, rapper, etc.) from either the lead, the categories attribute, or the description attribute. Still, we must keep in mind that one person might belong to more than one groups (i.e. have multiple jobs throughout his/her working life). 


## Blog Post 3: Final Report

### Introduction
It's rare to find a collection of human knowledge as rich and as well curated as the Wikipedia corpus. Yet, due to complexity of the natural language, computers cannot readily make use of this data. How do you make computers understand human language? How do you extract high-level information from text? Our group decided to tackle this challenge in hope of finding out new insights on Wikipedia pages.  

### Data Scraping and Cleaning
Our point of entry is https://en.wikipedia.org/wiki/Category:Births_by_century. This page contains links to every single Wikipedia article with a birth date listed. Scraping pages off it gives us a huge collection of links, 1.37 million in total, to Wikipedia articles about people. For two main reasons, we are only interested in the popular subset of our data:

<ol>
  <li>Popular articles are more relevant and are more representative of our perception of Wikipedia articles.</li>
  <li>Due to Wikipedia being a wiki, popular articles are better maintained, has higher quality, and less spelling and grammatical errors. This means that these articles are closer to our model of the English language.</li>
</ol>

We made a million requests to Wikipedia asking for page views of each article starting from the first day of 2018 and ending at the last. Below we give the top 20 most popular people pages and their view counts.

| Rank | Name              | Views    |
| ---: | :---------------- | :------- |
|    1 | Louis Tomlinson   | 32647670 |
|    2 | Freddie Mercury   | 22261530 |
|    3 | Elizabeth II      | 20049022 |
|    4 | Stephen Hawking   | 19025060 |
|    5 | Donald Trump      | 18663083 |
|    6 | Cristiano Ronaldo | 18467102 |
|    7 | Cardi B           | 17955598 |
|    8 | Elon Musk         | 17836456 |
|    9 | XXXTentacion      | 15249774 |
|   10 | Lionel Messi      | 13457818 |
|   11 | LeBron James      | 12555210 |
|   12 | Ariana Grande     | 12307383 |
|   13 | Jason Momoa       | 12208869 |
|   14 | 6ix9ine           | 12091008 |
|   15 | George H.W. Bush  | 12078037 |
|   16 | Anthony Bourdain  | 11867794 |
|   17 | Priyanka Chopra   | 11613801 |
|   18 | John McCain       | 11550433 |
|   19 | Queen Victoria    | 11415901 |
|   20 | Stan Lee          | 11352622 |

From this, we took the 10,000 most viewed pages as our subset. The cutoff at 10,000 is really quite arbitrary, but these top 10,000 articles are responsible for approximately half the total traffic.

<p align="center">
  <img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/page-views.png" style="width:600px;">
</p>

Now, it has never been a matter of debate that the most insightful part of a Wikipedia article is the so-called lead. In our case, the lead section tells us what the person does, what they are known for and, in general, why they deserve a Wikipedia article. While some pages have very short lead sections, most of the pages in the top 10,000 have considerably longer leads (since there are more to be said about popular people) than the rest of the corpus. Thus we were able to obtain all the data we hoped for. 

We clean the text for data exploration by normalizing case to lowercase, removing punctuation, filtering out stopwords, and lemmatizing whatever remains.

### Methodology
We are interested in the underlying patterns in our data, both in the types of pages that are popular, and in the underlying language structure of the data we collected. 

In order to explore the popular pages, we use various topic modelling techniques to find the groups of people whose Wikipedia page made it to the top 10,000 most viewed. To find the underlying language structure, we use LSTM, a recurrent neural network structure, to capture the underlying language style and generate a lead for a page on its own. 

### Data Exploration
<i>Hypothesis: Since Wikipedia lead commonly focuses on the person’s career path, any unsupervised clustering should result in clusters differentiated by occupations. </i>


#### Document Clustering with K-means Clustering
We also experimented with document clustering. We first turn Wikipedia pages to numeric representations using doc2vec. doc2vec is a modified word2vec algorithm, designed to capture semantic information of a document. As an addition to word2vec, a document ID is concatenated to the input word sequence. The learned embedding of each document ID is the vector representation of the document.

Once we obtain the representations of all the documents, we utilize the k-means algorithm to cluster the documents. We ran doc2vec and k-means on the dataset of 10,000 Wikipedia pages. Setting the number of clusters to 10, we obtained the following results with doc2vec.

<p align="center">
  <table style="width: 120%; border: 1px solid black; border-collapse: collapse; margin-left: -65px;">
      <tr style="border: 1px solid black">
          <th>Cluster</th>
          <th width="40%">Word Cloud</th> 
          <th>Example</th>
          <th>Number of Pages</th>
          <th>Variance</th>
        </tr>
        <tr style="border: 1px solid black">
          <td>1</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/clusters_n10_doc2vec/1.jpg" style="width: 250px; vertical-align: top"></td> 
          <td>LeBron James, Dwayne Johnson, Tonya Harding, Khabib Nurmagomedov, John Cena, Stephen Curry, Anthony Joshua, Caitlyn Jenner, Naomi Osaka, The Undertaker</td>
          <td>814</td>
          <td>0.0188</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>2</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/clusters_n10_doc2vec/2.jpg" style="width: 250px; vertical-align: top"></td> 
          <td>Anthony Bourdain, Meghan, Duchess of Sussex, Charles, Prince of Wales, Prince Philip, Duke of Edinburgh, Pablo Escobar, Jeff Bezos, Diana, Princess of Wales, Prince William, Duke of Cambridge, Jeffrey Dahmer, Prince Harry, Duke of Sussex</td>
          <td>956</td>
          <td>0.0188</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>3</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/clusters_n10_doc2vec/3.jpg" style="width: 250px; vertical-align: top"></td> 
          <td>Louis Tomlinson, Antonio Maria Magro, Dua Lipa, Pete Davidson, Andrew Cunanan, Hailey Baldwin, Dolores O'Riordan, John Paul Getty III, Lisa Bonet, Beto O'Rourke</td>
          <td>3597</td>
          <td>0.0055</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>4</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/clusters_n10_doc2vec/4.jpg" style="width: 250px; vertical-align: top"></td> 
          <td>Elon Musk, Stephen Hawking, P. T. Barnum, Albert Einstein, Ted Kaczynski, Steve Jobs, William Shakespeare, Bill Gates, Rajneesh, Nikola Tesla</td>
          <td>342</td>
          <td>0.0322</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>5</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/clusters_n10_doc2vec/5.jpg" style="width: 250px; vertical-align: top"></td> 
          <td>Donald Trump, George H. W. Bush, John McCain, Winston Churchill, Barack Obama, Adolf Hitler, Brett Kavanaugh, Mahatma Gandhi, George W. Bush, Martin Luther King Jr.</td>
          <td>336</td>
          <td>0.0429</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>6</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/clusters_n10_doc2vec/6.jpg" style="width: 250px; vertical-align: top"></td> 
          <td>Elizabeth II, Queen Victoria, Princess Margaret, Countess of Snowdon, George VI, Mary, Queen of Scots, Edward VIII, George V, Elizabeth I of England, Joaquín "El Chapo" Guzmán, Henry VIII of England</td>
          <td>253</td>
          <td>0.0428</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>7</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/clusters_n10_doc2vec/7.jpg" style="width: 250px; vertical-align: top"></td> 
          <td>Cristiano Ronaldo, Lionel Messi, Michael Jordan, Tom Brady, Kylian Mbappé, Mohamed Salah, Conor McGregor, Roger Federer, Virat Kohli, Neymar</td>
          <td>387</td>
          <td>0.0414</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>8</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/clusters_n10_doc2vec/8.jpg" style="width: 250px; vertical-align: top"></td> 
          <td>Cardi B, Freddie Mercury, XXXTentacion, Ariana Grande, 6ix9ine, Avicii, Donald Glover, Nick Jonas, Post Malone, Michael Jackson</td>
          <td>697</td>
          <td>0.0269</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>9</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/clusters_n10_doc2vec/9.jpg" style="width: 250px; vertical-align: top"></td> 
          <td>Jason Momoa, Stan Lee, Sylvester Stallone, Jennifer Aniston, Michael B. Jordan, Burt Reynolds, Ryan Reynolds, Chris Hemsworth, Josh Brolin, Gianni Versace</td>
          <td>1930</td>
          <td>0.0106</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>10</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/clusters_n10_doc2vec/10.jpg" style="width: 250px; vertical-align: top"></td> 
          <td>Priyanka Chopra, Sridevi, Tom Cruise, Demi Lovato, Clint Eastwood, Scarlett Johansson, Emily Blunt, Keanu Reeves, Bradley Cooper, John Krasinski</td>
          <td>688</td>
          <td>0.0247</td>
        </tr>
    </table>
</p>

Clustering allows us to better understand our dataset. From the table, we see that there are 6 main "categories" in our dataset: (1) athletes (Clusters 1, 7), (2) royalties (Clusters 2, 6), (3) actors (Clusters 3, 9, 10), (4) businessmen (Cluster 4), (5) politicians (Cluster 5), and (6) singers (Cluster 8). From the wordclouds, we also see that the important words are corresponding to each cluster. For instance, the word "work" was extracted as the most important word in Cluster 4, which represent those best known for their career successes such as Elon Musk, Albert Einstein and William Shakespeare. Moreover, we notice that the dataset is highly imbalanced as shown in the chart below. More than half of the dataset are actors/actresses or movies-related figures, while only 3% are political figures.

<p align="center">
 <img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/pie_kmeans.png" style="width:500px;">
</p>

However, there are some issues concerning our clustering. The first issue is a trade off between a large and small numbers of clusters. When we have too many clusters, some categories might be duplicated or very similar as the results we obtained when having 10 clusters. Yet, we found some interesting clusters such as one which mostly contains  Marvel celebrities including Ryan Reynolds, Stan Lee, and Jason Momoa, and another cluster which mostly contains football players. On the other hand, when we have a small number of clusters, a cluster may include irrelevant pages. As an example, Anthony Bourdain who was a chef was put in the royal families cluster (Cluster 2). While solving the trade off problem is not trivial, a possible work around for this is using alternative embeddings which better represent information about each cluster. As an idea, if we want to cluster documents based on occupations, we can build a neural network which embeds documents and is trained to predict occupations.

#### Document Clustering with Spectral Clustering
Seeing that K-Means clustering could not give us a perfect result, we try another clustering method: spectral clustering. After several trials of spectral clustering on doc2vec, tf-idf, and bag of words, we find that spectral clustering on bag of words gives the best result, which is shown below. 

<p align="center">
  <table style="width: 120%; border: 1px solid black; border-collapse: collapse; margin-left: -65px;">
      <tr style="border: 1px solid black">
          <th>Cluster</th>
          <th width="40%">Word Cloud</th> 
          <th>Example</th>
          <th>Number of Pages</th>
          <th>Variance</th>
        </tr>
        <tr style="border: 1px solid black">
          <td>1</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/spectralcluster_n10_wordcloud/1.png" style="width: 250px; vertical-align: top"></td> 
          <td>Elizabeth II, Freddie Mercury, Elon Musk, Stephen Hawking, Anthony Bourdain, Queen Victoria, Stan Lee, Princess Margaret, Countess of Snowdon,  Charles (Prince of Wales), Prince Philip (Duke of Edinburgh)</td>
          <td>987</td>
          <td>3.09e-09</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>2</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/spectralcluster_n10_wordcloud/2.png" style="width: 250px; vertical-align: top"></td> 
          <td>Antonio Maria Magro, Pete Davidson, Andrew Cunanan, Gianni Versace, Hailey Baldwin, Jeffrey Dahmer, Dolores O'Riordan, John Paul Getty III, Lisa Bonet, Chadwick Boseman</td>
          <td>3474</td>
          <td>4.60e-10</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>3</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/spectralcluster_n10_wordcloud/3.png" style="width: 250px; vertical-align: top"></td> 
          <td>Dwayne Johnson, Tonya Harding, Jeff Bezos, Khabib Nurmagomedov, Conor McGregor, Roger Federer, John Cena, Anthony Joshua, Serena Williams, Muhammad Ali</td>
          <td>377</td>
          <td>2.60e-09</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>4</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/spectralcluster_n10_wordcloud/4.png" style="width: 250px; vertical-align: top"></td> 
          <td>Donald Trump, George H. W. Bush, John McCain, Winston Churchill, Barack Obama, P. T. Barnum, Brett Kavanaugh, George W. Bush, John F. Kennedy, Abraham Lincoln</td>
          <td>602</td>
          <td>3.30e-09</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>5</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/spectralcluster_n10_wordcloud/5.png" style="width: 250px; vertical-align: top"></td> 
          <td>Tom Cruise, Demi Lovato, Scarlett Johansson, Emily Blunt, Bradley Cooper, Leonardo DiCaprio, Rami Malek, Ellen DeGeneres, Elizabeth Olsen, Robin Williams</td>
          <td>715</td>
          <td>1.55e-09</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>6</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/spectralcluster_n10_wordcloud/6.png" style="width: 250px; vertical-align: top"></td> 
          <td>LeBron James, Michael Jordan, Tom Brady, Stephen Curry, Shaquille O'Neal, Kobe Bryant, Nick Foles, O. J. Simpson, Kevin Durant, Patrick Mahomes</td>
          <td>507</td>
          <td>1.78e-09</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>7</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/spectralcluster_n10_wordcloud/7.png" style="width: 250px; vertical-align: top"></td> 
          <td>Cristiano Ronaldo, Lionel Messi, Kylian Mbappé, Mohamed Salah, Virat Kohli, Neymar, Harry Kane, Zlatan Ibrahimović, David Beckham, Pelé</td>
          <td>470</td>
          <td>2.15e-09</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>8</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/spectralcluster_n10_wordcloud/8.png" style="width: 250px; vertical-align: top"></td> 
          <td>Cardi B, Louis Tomlinson, XXXTentacion, Ariana Grande, 6ix9ine, Avicii, Donald Glover, Nick Jonas, Post Malone, Michael Jackson</td>
          <td>727</td>
          <td>2.32e-09</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>9</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/spectralcluster_n10_wordcloud/9.png" style="width: 250px; vertical-align: top"></td> 
          <td>Meghan, Duchess of Sussex, Jennifer Aniston, Kylie Jenner, Michael B. Jordan, Burt Reynolds, John Krasinski, Tom Hardy, Chris Pratt, Jennifer Lawrence, Fred Rogers</td>
          <td>1349</td>
          <td>1.07e-09</td>
        </tr>
        <tr style="border: 1px solid black">
          <td>10</td>
          <td><img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/spectralcluster_n10_wordcloud/10.png" style="width: 250px; vertical-align: top"></td> 
          <td>Jason Momoa, Priyanka Chopra, Sridevi, Sylvester Stallone, Clint Eastwood, Ryan Reynolds, Chris Hemsworth, Josh Brolin, Keanu Reeves, Mila Kunis</td>
          <td>792</td>
          <td>1.62e-09</td>
        </tr>
    </table>
</p>

Below is the pie chart showing the percentage of people in each career grouped by the model.

<p align="center">
 <img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/pie_spectral.png" style="width:500px;">
</p>

Notice that spectral clustering on bag of words is better than k-means clustering on doc2vec in categorizing athletes who compete in different sports and actors who are known for different types of media (i.e. TV show or movie). Nevertheless, our spectral clustering cannot capture other career groups apart from athletes, actors, singers, and politicians, while our k-means clustering model can group royalties and businessmen together. This observation has not taken into account the accuracy of each models, which we will be discussed further later in this report.

As can be seen above, these two methods of clustering supports our hypothesis that unsupervised clustering method should give us clusters with different occupations. In addition, as it is shown in the pie charts that the majority of people in our dataset are actors/actresses, this bias potentially affects our training process (e.g. many pages fall into the “Actors” category.) Although we can tackle this by including more pages from minority classes, this method might not entirely solve the problem as our dataset lacks high-quality labels. We will discuss our attempts to generate labels later in the Additional Work section. 

### Document Generation
As our dataset is imbalanced, we are motivated to work on unsupervised tasks. We trained a word-based language model to capture the underlying structure of the leads of the Wikipedia pages. Our model is constructed by an embedding layer, two cells of long short-term memories (LSTMs), and a linear layer as shown in the diagram below. Outputs of the network are probabilities for predicting the next words. We train the network to optimize the cross entropy loss between actual next words and predicted words. 

<p align="center">
 <img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/network.png">
</p>

In the preprocessing step, we filtered for only those with more than 100 tokens (which turns out to be a little over half.)  Uncommon words were replaced by their part of speech using a different pre-trained neural network tagger. We also converted all words to lower cases and split punctuations with space.

To generate the text, we softmaxed the last layer of the network, used the output as the word transition kernel and sampled from the Markov chain. After we trained for 50,000 samples each of window size 1,000 on the top 100,000 lead sections, the network could learn to keep gender pronouns consistent (using she for females) within the same sentence, but not within one section. Using the recurrent neural network, we were able to generate texts such as the following:

<p style="font-family: courier; background: #fffadd">
  anne PROPN PROPN ( born december NUM , NUM ) is an american singer - songwriter . she has fought several small - scale roles , including the title role in the films PROPN PROPN in chicago , VERB on hell ( NUM ) and the PROPN games ( NUM ) . his films gained notoriety for her ADJ stage persona - NOUN in the movie PROPN ( NUM ) , PROPN PROPN 's PROPN in tamil and juliet kapoor ( NUM ) and le ORG in PROPN ( NUM ) and is shown in the south korea NOUN movement , ADJ PROPN PROPN PROPN , and the PROPN PROPN instructor . PROPN made her debut in NUM for PROPN PROPN in NUM , was soon discovered in the telugu film al PROPN , in india 's NOUN in telugu . she married NUM children in his own reality show , PROPN PROPN ( NUM ) , as well as his long - running comedy NOUN , PROPN ( NUM ) .
</p>

<p style="font-family: courier; background: #fffadd">
  louise PROPN ( born december NUM , NUM ) is an american country music singer - songwriter , musician , and actress . she has received a number of NUM awards , and an academy award nomination and a grammy award for best featured actress in a musical . she is best known for her roles on the nbc sitcom the dead . she is also the co - creator of ORG ORG &amp; ORG . in NUM PROPN became the highest - paid cast member on season NUM of the big brother : NOUN .
</p>

Due to the skewed dataset, our model likely generated paragraphs about celebrities as shown above. However, it could also generate a paragraph about a professor as follow:

<p style="font-family: courier; background: #fffadd">
  joseph edward PROPN " PROPN " PROPN iii ( NUM march NUM – NUM october NUM ) was a german philosopher , professor , NOUN , NOUN , journalist , and NOUN who supported the discipline of NOUN and on developing a formal newspaper , such as the NORP PROPN ( NUM ) and the PROPN PROPN of PROPN ( NUM ) . he studied at the centre of the arts in new york university , where he worked as a teacher , who was an assistant professor in the department of philosophy at the massachusetts institute of technology . before entering politics , he emigrated to new york city as a teacher and NOUN . he studied under the new york ORG school . a graduate of harvard university , ORG is a graduate of the secretary of fine review , PROPN PROPN and PROPN holmes , in which he has been a model and journalist . NORP served as a lawyer of the ORG society of canada .
</p>

### Additional Work

#### Label Extraction from ‘Description’ Property 
One of the biggest challenges of the Wikipedia dataset is the lack of high-quality labels. In this section, we describe our attempts to automatically extracting occupation titles from Wikipedia pages. Since all Wikipedia pages do not share a fixed format, we cannot simply scrape an occupation title from the text. Some pages have an infobox summarizing the biography of a person. However, not all infoboxes include occupations and some pages have different formats. 

Our first attempt was to to automatically extract occupation titles from the "description" property of Wikipedia pages using a pre-trained word2vec model. To be clear, some examples of description properties are "Italian contemporary artist", "Lesotho politician", and "Chinese print artist". At this point, you might have noticed that descriptions are very close to what we want, but some adjectives need to be removed. We then compute the "occupation-likeliness" score of each word in description using cosine similarity of the word's embedding and the embedding of the word "job" or "occupation". Additionally, most adjectives are related to nationalities. So we subtract the score with cosine similarity of words like "nationality" or "country" as well.

Below shows examples of page descriptions and the extracted occupations.

Good examples:

<p align="center">
  <table style="width:100%">
    <tr>
      <th>Description</th>
      <th>Extracted Labels</th> 
    </tr>
    <tr>
      <td>Bangladeshi physicist</td>
      <td>'physicist'</td> 
    </tr>
    <tr>
      <td>Tongan politician</td>
      <td>'politician'</td> 
    </tr>
    <tr>
      <td>United States rapper</td>
      <td>'rapper'</td> 
    </tr>
    <tr>
      <td>anonymous writer</td>
      <td>'writer'</td> 
    </tr>
    <tr>
      <td>Current Mayor of Chittagong City Corporation</td>
      <td>'mayor'</td> 
    </tr>
  </table>
</p>

Bad examples:
<p align="center">
  <table style="width:100%">
    <tr>
      <th>Description</th>
      <th>Extracted Labels</th> 
    </tr>
    <tr>
      <td>New Zealand drifting race driver</td>
      <td>'new', 'race', 'driver'</td> 
    </tr>
    <tr>
      <td>American record producer</td>
      <td>'record'</td> 
    </tr>
    <tr>
      <td>professional wrestler</td>
      <td>'professional'</td> 
    </tr>
    <tr>
      <td>Olympic judoka</td>
      <td>'olympic'</td> 
    </tr>
    <tr>
      <td>Baritone saxophone player based in New York City</td>
      <td>'baritone', 'saxophone', 'player', 'york'</td> 
    </tr>
  </table>
</p>

While this method of labelling can give us unlimited choices of labels, the disadvantages are:

<ol>
  <li>Two different labels, for example ‘singer’ and ‘rapper’, might be able to be grouped into one. Since there could be unlimited choices of labels generated but a very limited ways to cluster, the amount of work to group labels together is tremendous.</li>
  <li>The implementation cannot perfectly extract words of occupations as can be seen in bad examples.</li>
  <li>Some occupation nouns in the description cannot represent the category a person should fall into. For example, the word ‘player’ might be seen to belong to ‘sport’ category although it might come from a description about a ‘saxophone player’.</li>
  <li>432 pages do not have description, which accounts for 0.432% of our dataset. </li>
</ol>

#### Label Generation by Word Detection
Knowing that our models groups people into at most 6 categories -  athletes, actors, politicians, singers, royalties, and businessmen - but label generation method above could give us numerous distinct labels, we think it might not be the best way considering time and effort we need to clean the labels to match with our 6 categories from the clustering. Therefore, we decide to go the other way around on creating labels that match these categories. That is, we come up with words that are unique to each category. To illustrate, word including ‘actor', 'actress' ,'film', 'television', 'oscar', 'comedian', 'show', 'role', 'character', 'host', 'cast', and 'serie', we believe, should appear only if that person’s career is heavily related to being an actor/actress. With those words, we detect how many time they appear in the lead of a person, and we label that person based on the category that has the highest number of related words. Notice that there are people that do not have any related words in any given categories. Thus, we have them fall into ‘None’ category. 

This method can only be done if we already have a clustering result that every cluster can be easily named the similarity among its group and its uniqueness. 

With this implementation on actors, singers, politicians, athletes, businesspeople, and royalties, we obtain the following percentage of people in each occupation. 

<p align="center">
 <img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/pie_detect.png" style="width:500px;">
</p>

To compute the accuracy of our clustering model based on labels obtained from word detection, we use the following equations.

<p align="center">
 <img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/accuracy_eq_.png" style="width:700px;">
</p>

In the other words, the accuracy within the cluster tells us how robust each cluster is while the accuracy of the label type indicates how good the cluster can group all pages with a particular label.

With this implementation on athletes, actors, politicians, singers, royalties, and businessmen categories, our k-means clustering on doc2vec gives us 67.03% total accuracy. 

<p align="center">
  <table style="width:100%">
    <tr>
      <th>Career</th>
      <th>Accuracy in the cluster</th> 
      <th>Accuracy of the label type</th>
    </tr>
    <tr>
      <td>Actors</td>
      <td>70.83%</td> 
      <td>93.22%</td> 
    </tr>
    <tr>
      <td>Businesspeople</td>
      <td>22.19%</td> 
      <td>17.69%</td> 
    </tr>
    <tr>
      <td>Politicians</td>
      <td>87.33%</td> 
      <td>42.68%</td> 
    </tr>
    <tr>
      <td>Royalties</td>
      <td>20.86%</td> 
      <td>69.85%</td> 
    </tr>
    <tr>
      <td>Singers</td>
      <td>93.94%</td> 
      <td>46.48%</td> 
    </tr>
    <tr>
      <td>Athletes</td>
      <td>98.94%</td> 
      <td>50.51%</td> 
    </tr>
  </table>
</p>

Similarly, using word detection model, we get that our spectral clustering model on bag of words achieves 71.73% total accuracy. Notice that the careers that appear in this clustering are only actors, politicians, singers, and athletes. 

<p align="center">
  <table style="width:100%">
    <tr>
      <th>Career</th>
      <th>Accuracy in the cluster</th> 
      <th>Accuracy of the label type</th>
    </tr>
    <tr>
      <td>Actors</td>
      <td>71.39%</td> 
      <td>95.68%</td> 
    </tr>
    <tr>
      <td>Politicians</td>
      <td>68.97%</td> 
      <td>65.88%</td> 
    </tr>
    <tr>
      <td>Singers</td>
      <td>81.84%</td> 
      <td>50.24%</td> 
    </tr>
    <tr>
      <td>Athletes</td>
      <td>94.76%</td> 
      <td>79.37%</td> 
    </tr>
    <tr>
      <td>Others</td>
      <td>51.78%</td> 
      <td>60.78%</td> 
    </tr>
  </table>
</p>

With these accuracy statistics, we might conclude that k-means clustering gives more robust clusters (i.e. fewer errors in each cluster), while clusters from spectral clustering can better represent each career type.

Another observation from these statistics is that while k-means clustering can group royalties and businesspeople, the clusters representing them have very low accuracy level compared to the other clusters.

#### Similarity Search: Nearest Neighbor Search with 'Categories' Property
From 10,000 most viewed Wikipedia pages in People category, 9,620 pages has 'Categories' property that can be accessed by requesting to Wikipedia API.

To illustrate, Wikipedia has 'Michael Jackson' fit the following categories: '1958 births', '2009 deaths', '20th-century American singers', '21st-century American singers', 'AC with 15 elements', 'Accidental deaths in California', 'African-American choreographers', 'African-American dancers', 'African-American male dancers', 'African-American male singers'.

Since 'Categories' property summarizes each person and focus on their occupations, nationalities, and ethnicities, we hope that we can classify people into groups using their categories already made by Wikipedia.

Firstly, we tried Nearest Neighbor Search to find the closest match of each person. In the data preparation process, we discard categories that do not relate to the person's identity. Those categories commonly contains the words 'articles', 'pages', etc. Then, we use a one-hot encoding i.e. dummy variable to convert categories into integer data. After the preparation, we have sklearn.NearestNeighbor fit the one-hot data and predict top two closest match of each person.

In the first try, we have a dummy variable for every category that appears more than 7 times. The predictions of the 17 most viewed Wikipedia pages are as follow.

<p align="center" style="margin-left: -60px;">
 <img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/untokenized.jpg">
</p>

To improve the predictions, we tokenize the categories and have important words representing each person instead. This is because we can create subcategories from the categories given by Wikipedia. For instance, with tokenized version, Michael Jackson will belong to 'American', 'African-American', 'dancers', 'singers', etc. Without the tokenization, Michael Jackson would never be matched to an American dancer even though they are similar.

Thus, we tokenize important categories and perform Nearest Neighbor Search with the new one-hot matrix. The predictions from this try is more accurate.

<p align="center" style="margin-left: -60px;">
 <img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/tokenized.jpg">
</p>

#### Topic Modeling: Latent Dirichlet Allocation
We also experimented with Latent Dirichlet Allocation for soft assignment of topics. The top words for each of the ten topics we got are as follow:

<p align="center">
 <img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/lda_res.png" style="width:700px;">
</p>

We can see that this is pretty similar to the results we got from other types of clustering. 

#### Keyword Extraction: Term Frequency-Inverse Document Frequency
Altogether, the 10000 lead sections contain 5461 unique words. To quantify the importance of each word, we use a popular keyword extraction technique called term frequency-inverse document frequency (tf-idf.)

The main insight of tf-idf is that important words should give information about the articles containing them. In particular, they should appear rather frequently in documents that contain them, but not so ubiquitously that they show up in every other article. The simplest of the tf-idf formulae articulating this train of thought is

<p align="center">
 <img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/tfidf_eq.png" style="width:500px;">
</p>

Using the tf-idf formula above to rank the words gives us an expected result, which we show in the wordcloud below.

<p align="center">
 <img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/wordcloud_bad_names.png">
</p>

We can see that the most important words are names, which makes sense because important words should distinguish between articles. But not very interesting because we already know that these words are important. To counteract this phenomenon, we only consider common English words, i.e., words you would see in an English dictionary. This leaves us with 3815 unique words, the wordcloud of whose is shown below.

<p align="center">
 <img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/wordcloud_bad_words.png">
</p>

Words like “cave” and “pompey” and “vaccine” definitely tell us a lot about the articles containing them, but they don’t cover enough of the corpus to give us a big picture. (If you read on, you will see that over half of the articles are about actors or actresses, but it’s impossible to tell just looking at these words that there are any famous actors.) Upon closer inspection, these words only appear in very few documents, so their idf-weights are very high despite not appearing a lot. We experimented with other tf-idf formula:

<p align="center">
 <img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/wordcloud_alternate1.png">
</p>

After a few different experiment with weighting, we ended up with a method to calculate idf which wikipedia calls probabilistic idf.

<p align="center">
 <img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/idf_eq.png" style="width:250px;">
</p>

where $n_w$ is the number of articles containing the term $w$.

<p align="center">
 <img src="https://raw.githubusercontent.com/PanthonImem/CS1951a-BlogPost/master/Photos/wordcloud.png">
</p>

[blog-1](blog-1.md)
[blog-2](blog-2.md)
[blog-3](blog-3.md)
