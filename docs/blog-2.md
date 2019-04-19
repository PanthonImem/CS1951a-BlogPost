{% include mathjax.html %}

# CS1951A Final Project

## Blog Post 2
### Document Clustering: Spectral Clustering with Bag of Words 

To improve the performance of document clustering, we tried clustering socuments with [spectral clustering](https://en.wikipedia.org/wiki/Spectral_clustering). Instead of grouping points that are close to one another together like K-means approach, spectral clustering clusters points that connect to one another together. With many feature extraction methods including Doc2Vec and TF-IDF, spectral clustering with bag of words gives the best result. Unlike the result from Kmeans with Doc2Vec, this result groups athletes who play different sports. However, there is no group of royalty in this clustering result. 

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
          <td>Cristiano Ronaldo, Lionel Messi, Kylian Mbappé, Mohamed Salah, Virat Kohli, Neymar, Harry Kane, Zlatan Ibrahimovi%C4%87, David Beckham, Pelé</td>
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

Compared two clusterings with the pie chart using keywords from categories attribute, we can see that the spectral clustering with bag of words is better at clustering actors/actresses and singers, while kmeans clustering with Doc2Vec is better at grouping athletes and politicians. 

The pie chart obtained from keywords in categories is however not completely accurate itself because, firstly, one person may have more than one occupation indicated in the category property; secondly, Wikipedia only give 9,620 pages from 10,000 most viewed Wikipedia articles in people the category property; and lastly, to compare with the results from k-mean algorithm, we disregarded several jobs, including novelists, comedians, and artists. 

To further develop the accuracy test, we should create labels for each person. One possible method is to detect a keyword (i.e. actor, actress, government, politician, singer, rapper, etc.) from either the lead, the categories attribute, or the description attribute. Still, we must keep in mind that one person might belong to more than one groups (i.e. have multiple jobs throughout his/her working life). 
