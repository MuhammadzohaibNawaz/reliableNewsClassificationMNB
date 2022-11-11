import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
data = pd.read_csv("dataset.csv")

def handle_nan(data):
    '''Input: Data to the function containing Nan values.
       Output : Cleaned data containing no Nan values.
       Function: Cleaning Nan values.
     '''
    data = data.fillna(" ")
    return data

data = handle_nan(data)
X=data.iloc[:,:-1]
y=data.iloc[:,-1]

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
data=data.dropna()
#since null value rows are dropped indexes needs to be reset
messages=data.copy()
messages.reset_index(inplace=True)

# Below steps are performed to preprocess the data
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,ngram_range=(1,3))
X = cv.fit_transform(corpus).toarray()

y=messages.iloc[:,-1]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.24, random_state=0) #train/test splitting is performed here

import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    sns.heatmap(cm)
    # print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB() #multinomial naive bayes is selected as classifier

from sklearn import metrics
from sklearn.metrics import classification_report,accuracy_score
import numpy as np
import itertools


classifier.fit(X_train, y_train) #model is trained here
pred = classifier.predict(X_test) #trained model is saved here
score = metrics.accuracy_score(y_test, pred) #accuracy of trained MNB classifier is measure against test set
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL']) #confusion matrix is plot here
cr1    = classification_report(y_test,pred) #classification matrices are calculated here
print(cr1) #classification measures are displayed here
