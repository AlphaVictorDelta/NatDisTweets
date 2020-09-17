# Modules used
import pandas as pd
import string
import re
import sklearn   
import nltk

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 

nltk.download('wordnet')
nltk.download('words')


def remove_non(sent):
    words = set(nltk.corpus.words.words())
    #sent = "Io andiamo to the beach with my amico."
    sent = " ".join(w for w in nltk.wordpunct_tokenize(sent) if w.lower() in words or not w.isalpha())
    return sent

# Text maker Function
def concat(text):
    # text is the form of a list
    s = ' '
    result = s.join(text)
    return result

# Lemmatizer Function
def lemmatizer(text):  
    wnl = nltk.WordNetLemmatizer()
    return [wnl.lemmatize(word) for word in text]

def stemmer(text):
	ps = PorterStemmer() 
	return [ps.stem(word) for word in text.split()]

# cleaner Function
def cleanText(textRecords):

	#return this 
    count = 0
    cleanedText = []
    
    for text in textRecords:
    
        # regex to remove all and any links
        URL_REGEX = r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"      
        result = re.sub(URL_REGEX, '', text, flags=re.MULTILINE)   
        
        # Remove Numbers 
        result = ''.join([i for i in result if not i.isdigit()])
        
         # to upper
        result = result.lower() 
    
        # make list of words
        words = result.split() 

        # removes all puntuation marks
        #table = str.maketrans('', '', string.punctuation)          
        #stripped = [w.translate(table) for w in words]

 		 # remove this character and empty spaces from the list
        stripped = words
        for i in stripped:     
            if '\x89Û' in i or ' ' in i or '&amp;' in i:
                stripped.remove(i)  
                
                
        # retain root words
        stripped = lemmatizer(stripped)
        
        # remove non english words
        prefinal = remove_non(concat(stripped))

        # stem words
        #final = stemmer(prefinal)

        # data output
        count += 1
        print(count)  # Runs Real slow, to keep track of progress
        print(prefinal)
        
        # Concatenate strings to form cleaned text
        cleanedText.append(concat(prefinal))

    return cleanedText

# clean Training data
disasterTrain = pd.read_csv('train.csv')
disasterTrain.drop(columns = ['keyword', 'location'], inplace=True)
textRecords_train = disasterTrain['text']
cleanRecords_train = cleanText(textRecords_train)
cleanRecords_train = pd.Series(cleanRecords_train)
disasterTrain['CleanedText'] = cleanRecords_train
disasterTrain.drop(columns = 'text', inplace=True)
print(disasterTrain) 

# clean Testing data
disasterTest = pd.read_csv('test.csv')
disasterTest.drop(columns = ['keyword', 'location'], inplace=True)
textRecords_test = disasterTest['text']
cleanRecords_test = cleanText(textRecords_test)
cleanRecords_test = pd.Series(cleanRecords_test)
disasterTest['CleanedText'] = cleanRecords_test
disasterTest.drop(columns = 'text', inplace=True)
print(disasterTest)

# Vectorise Text(train) input!
count_vectorizer = feature_extraction.text.CountVectorizer(strip_accents = 'ascii',stop_words = 'english')
disasterVectors_train = count_vectorizer.fit_transform(disasterTrain["CleanedText"])
tfidf_trnsfrmr = TfidfTransformer(use_idf=False)
tfidf_train = tfidf_trnsfrmr.fit_transform(disasterVectors_train)
disasterVectors_test = tfidf_trnsfrmr.transform(count_vectorizer.transform(disasterTest["CleanedText"]))
disasterVectors_test = count_vectorizer.transform(disasterTest["CleanedText"])

# Print data parameters after cleaning and Transforming
print(disasterVectors_train[0].todense().shape)
print(disasterVectors_train[0].todense())
print(disasterVectors_test[0].todense().shape)
print(disasterVectors_test[0].todense())


# Save dataframes locally to avoid running this code again and again
disasterTrain.to_csv (r'C:\Users\Akshay Desai\Desktop\All Projects\Real or Not\disasterTrain.csv', index = False, header=True)
disasterTest.to_csv (r'C:\Users\Akshay Desai\Desktop\All Projects\Real or Not\disasterTest.csv', index = False, header=True)