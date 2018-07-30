import time
import stomp

# Connection properties for subscribing queue and publish queue
user='admin'
password = 'admin'
host = 'localhost'
port = '61613'
destination ='TestQ'

user_send='admin'
password_send = 'admin'
host_send = 'localhost'
port_send = '61613'
destination_send ='ReceiveQ'

# machine learning algorithm will classify the incoming text into the following and suggest next best action
sentiment_categories = ['positive','negative']
next_best_action = ['Offer Discounted Ancillaries','Customer Attrition Risk. Offer Seat Upgrade in next Booking or Transfer Call']

# class for message listener - consuming the incoming messages
class MyListener():
    
    # on_message method will be called when source system sends the notes/message to be classified
    # this method will read the message, vectorize it and apply machine learning algorithm to classify and send it to queue for further consumption
    def on_message(self, headers, message):
        print("Received %s" % message)
        docs_new = []
        docs_new.append(message)
        print(docs_new)
        # Vectorize the text
        X_new_counts = count_vect.transform(docs_new)
        # calculate tf-idf
        X_new_tfidf = tf_transformer.transform(X_new_counts)
        # classify the incoming text into one of the predefined categories using Multinomial Naive Bayes supervised learning
        predicted = clf.predict(X_new_tfidf)
        print('Multinomial NB Prediction is %r' % sentiment_categories[predicted[0]])
        #vectorize using stemmed vectorizer from nltk
        X_new_counts = stemmed_count_vect.transform(docs_new)
        # calculate tf-idf for the incoming text/message/notes
        X_new_tfidf = tf_transformer_nltk.transform(X_new_counts)
        # predict/classify the category using multinomial naive bayes
        predicted1 = clf1.predict(X_new_tfidf)
        # send the data to activemq queue to be consumed by the web socket.
        data = "Sentiment is "+sentiment_categories[predicted1[0]]+" "+next_best_action[predicted1[0]]
        time.sleep(0.1) 
        conn.send(destination_send, data)#send to queue
        
        print('Multinomial NB NLTK Prediction is %r' % sentiment_categories[predicted1[0]])
        #predict/classify incoming message/text/notes using simple vector machine
        predicted2 = clf2.predict(X_new_tfidf)
        print('SGD NLTK Prediction is %r' % sentiment_categories[predicted2[0]])
        
        
        
A=[]
B=[]
# the file test_data.txt consists of training data. the 1st column is class/category and 2nd column onwards is the message/notes
#f = open("C:/rasa/test_data.txt")
f = open("test_data.txt")
x= f.read().split('\n')
for text in x:
    A.append(int(text.split('\t')[0]))
    B.append(text.split('\t')[1])
    
f.close()
#print(A)
#print(B)

# vectorize the input texts. create number representation of the text.
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X=count_vect.fit_transform(B)

# calculate 'term frequency'-'inverse document frequency' for each word in each text message
# this will help in setting up relative weight for the words in each document/text-message
from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer_in = TfidfTransformer()
tf_transformer = tf_transformer_in.fit(X)
X_train = tf_transformer.transform(X)

# train the model to classify/predict based on Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train, A)

# use NLTK to improve upon prediction ruling out stopwords and using stemming  
import nltk
#nltk.download('popular')
nltk.download('stopwords')
from nltk.stem.snowball import EnglishStemmer
stemmer = EnglishStemmer()
analyzer = CountVectorizer().build_analyzer()
def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

stemmed_count_vect = CountVectorizer(analyzer=stemmed_words)
X=stemmed_count_vect.fit_transform(B)
tf_transformer_in_nltk = TfidfTransformer()
tf_transformer_nltk = tf_transformer_in_nltk.fit(X)
X_train = tf_transformer_nltk.transform(X)

clf1 = MultinomialNB().fit(X_train, A)

from sklearn.linear_model import SGDClassifier
clf2 = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None).fit(X_train, A)
    
# prepare connection based on stomp for receiving messages        
conn = stomp.Connection(host_and_ports = [(host, port)])
conn.set_listener('my_listener', MyListener())
conn.start()
conn.connect(login=user,passcode=password)
conn.subscribe(id='1234', destination=destination, ack='auto')
# prepare connection based on stomp for sending classification result and next-best-action
conn_send = stomp.Connection(host_and_ports = [(host_send, port_send)])
conn_send.start()
conn_send.connect(login=user_send,passcode=password_send)


print("Waiting for messages...")
# sleep for 10 miliseconds
while 1: 
    time.sleep(10) 