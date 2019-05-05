# NLTK
Steps involved in implementing NLTK
## Step 1: Importing necessary libraries
import re
import nltk
nltk.download("all")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

In the above re package is for data cleaning and review.  Nltk is the main package.  

## Step 2: Cleaning and reviewing data
Suppose you have a dataset full of 1000 rows then this is how you go about it:

First you initialize an empty list called corpus.  Then you create a loop to read through all rows and clean them. 
As you go about cleaning them one.  In the first step you remove all characters apart from a-z and A-Z.  Remaining characters you replace it with spaces.  Also you mainly take in the 
Review column here which has the verbiage of the reviews. 

Next you change all characters to lower case
Next you split each sentence into a list of words
Next you do stemming where you get to the stem word for each word
Then you apply stopwords to remove all the general article words like is, if, to etc.
Next you join all the words of each row into a sentence separated by a space
Next you populate the corpus list sentence by sentence

## Step 3 : Create a Bag of Words
Creating the Bag of Words model
A bag of words is a table matrix where each review corresponds to a row and each column corresponds to a unique word across all 
reviews together.  The cells will have a number which corresponds to the number of times a word appears  in that particular
review.  Therefore in this table you will see a lot of cells with zeroes.  So a matrix with a lot of zeroes in it is called
as a SPARSE MATRIX.  This entire process of creating a SPARSE MATRIX is called TOKENIZATION.
Also note that CountVectorizer (below) can do all of the data cleaning that re can do above.  However its better to use re
in the below step in a step-by-step fashion
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
X.shape
Out[9]:
(1000, 1565)

Essentially you use the process of TOKENIZATION to create a bag of words.  A Bag of Words is a table matrix where each review corresponds to a row and each column corresponds to a unique word across all reviews together.  The cells will have a number which corresponds to the number of times a word appears  in that particular review.  Therefore in this table you will see a lot of cells with zeroes.  So a matrix with a lot of zeroes in it is called as a SPARSE MATRIX.  
This entire process of creating a SPARSE MATRIX is called TOKENIZATION.
 
Here we see the number of unique words above turns out to be 1565.  Out of these there could be a number of useless words like names of individuals etc.  Here you change the 
Above code in this manner:
### The above X.shape reveals that there are 1000 rows (something we already knew as there were 1000 reviews) and 1565 columns (one
### column for each word).  Some words may be unimportant words.  Hence its safe to choose 1500 columns/features.  So we redo the
### code like this
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
Now in the above step - X becomes the list of features or the independent variables
So here below we go back and check the values of the original dataset.  
ds.values

In the next step we would be interested only in the second column i.e. wether a review is positive or not.  This is the label
or the dependent variable
y = ds.iloc[:, 1].values

## Step 4: Prepare the data for training
Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


## Step 5: Do the actual training
The three most common algos used for NLP are Naive Bayes, Decision Tree Classification and Random Forest Classification. 
CART, C5.0, Maximum Entropy:  CART or Classification And Regression Trees are a broad category of models., not one specific 
model. Also C5.0 is a methodology used in applying Decision trees, and is one of the most common approaches. 
Maximum Entropy is a method that multiple models use to establish decision trees and model learning. 
Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

Predicting the Test set results
y_pred = classifier.predict(X_test)

## Step 6: Measure the metrics
Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

