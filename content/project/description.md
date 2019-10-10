+++

title =  "Kict It Up!"

# [[gallery_item]]
# album = "Screenshots"
# image = "Execution speeds.jpg"
+++
# The dataset used for this has been taken from [Kaggle](https://www.kaggle.com/jessicali9530/kuc-hackathon-winter-2018/home).
# This dataset consists of patient reviews for numerous drugs that have been prescribed for the symptoms that they have experienced, and how effective they have been.

# The code was coded initially on Jupyter Notebook to have a stepwise visual response of the code flow.
The code was finalized on Notepad++.

Any other text editor will do just fine as long as they support the Python libraries that are being used.

For the application, Flask was used.

To host the program on the server, ngrok.exe was used to host the web app.


**<h2> Challenges faced:</h2>**

1. Initially the dataset was really big, thus it was taking time to read the entire dataset and then perform the calculation. The time taken was coming around 6-7 seconds, which is a terrible figure for a search module. Hence, the decision was made to initialize the reading of the dataset file at the beginning of the search functioning, thus making it easier to fetch the data rather than reading it continuously for every search iteration.

2. For a big dataset, with more than 100k entries, acquiring the posting lists is a big task.<br>

**<h2>Improvements:</h2>**

1. Elimination of stop words and lemmatizing existing words.

2. Passing the functions through a class, to make data retrieval from the dataset quicker, making search results faster.

3. Extraction of unique words while creating the word bank, thus reducing redundancy.
