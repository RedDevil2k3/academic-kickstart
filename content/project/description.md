+++

title =  "Kict It Up!"

# [[gallery_item]]
# album = ""
# image = ""
+++

**<h1> Information at your fingertips </h1>**

The world today is driven by information and knowledge and searching for things online has become such an integral part of lives. Google and Wikipedia quenches our thirst for knowledge nearly on a daily basis and people love to use them since they are extremly fast and boringly easy. Similarly, here I am trying to combine the features of the two in-order to build something specificly better concentrating majorly on Football aka SOCCER.
Inorder to have enough free text data to satisfy a "free text query search" I had to crawl up tonnes of relevant pages from WIKIPEDIA (which ofcourse serves a the major souce for free text). As humans, we love stats since they help us rank things so for the purpose of ranking an classifying data I looked up to sources like DATAWORLD.IO and kaggle for some more numerically intense datasets.



**<h2> Data Pre-processing </h2>**

Crawling up data from wikipedia articles doesn't render a very clean outcomes and hence leaves us with a messy soup of words and what-not. So, data pre-processing and fine-tuning becomes the most important first step before starting to build-up our DB.

Pre-processing involves several steps like:

i. Converting everything to small-caps.
ii. Removing additional white-spaces.
iii. Getting rid off commas, prepositions and known stop words.
iv. Removing duplicity.

Once, cleaned this data could be used to build up dictionaries for further uses.



**<h2> tf-idf </h2>**
  
Term Frequency : the number of times a term occurs in a document is called its term frequency.It can be calculated as : (frequnecy of a word in a document) / (total no. of words in that document)


Inverse Document Frequency : inverse document frequency diminishes the weight of terms that occur very frequently in the document set and increases the weight of rare terms.It can be calculated as:
(total no. of terms in the document corpus) / (no. of docs in which term 't' appears)



**<h2> Challenges faced</h2>**

1. Initially the dataset grew too big and very specific while scrapping uo the wikipedia since I was more       focussed on just one particular football club "Real Madrid FC" which lead to problems like really important terms were ranked approx 0 idf.

2. Too many unrelevant anchor tags were accessed during the first phase of crawling and it became very difficult to make sense of relevance between certain docs.

3. Specificity of data resulted in poor or strict search queries.



**<h2> Improvement </h2>**

1. To tackle the problem rised due to specificity of data I decided to increase the range of topics from just one club to the whole league. As a result now there's a better bank of varied and more distinct words.

2. Elimination of duplicate words previously would reduce the data size by half but after diversifying the datasets it works as intented since there are more distinct terms available.



**<h2> References </h2>**

https://nlp.stanford.edu/IR-book/pdf/02voc.pdf
https://nlp.stanford.edu/IR-book/pdf/03dict.pdf
https://nlp.stanford.edu/IR-book/pdf/06vect.pdf
https://github.com/Simon-Fukada
http://blog.josephwilk.net/projects/building-a-vector-space-search-engine-in-python.html
