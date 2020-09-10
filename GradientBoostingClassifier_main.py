# Modules used
import pandas as pd
import string
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

# Text maker Function
def concat(text):
    # text is the form of a list
    s = ' '
    result = s.join(text)
    return result

# cleaner Function
def cleanText(textRecords):
    count = 0
    #return this 
    cleanedText = []

    for text in textRecords:
    
        # regex to remove all and any links
        URL_REGEX = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:\'\".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""
        result = re.sub(URL_REGEX, '', text, flags=re.MULTILINE)   
        
        # Remove Numbers 
        result = ''.join([i for i in result if not i.isdigit()])
        
         # to upper
        result = result.upper() 
    
        # make list of words
        words = result.split() 
    
        # removes all puntuation marks
        table = str.maketrans('', '', string.punctuation)          
        stripped = [w.translate(table) for w in words]
    
        count += 1 # keeping track of number of conversions
    
        # remove this character and empty spaces from the list
        for i in stripped:     
            if '\x89Û' in i or ' ' in i:
                stripped.remove(i)  
            
        # Concatenate strings to form cleaned text
        cleanedText.append(concat(stripped))
    print(count)
    return cleanedText

# clean Training data
disasterTrain = pd.read_csv('train.csv')# load File
disasterTrain.drop(columns = ['keyword', 'location'], inplace=True)

textRecords_train = disasterTrain['text']
cleanRecords_train = cleanText(textRecords_train)
cleanRecords_train = pd.Series(cleanRecords_train)
disasterTrain['CleanedText'] = cleanRecords_train
disasterTrain.drop(columns = 'text', inplace=True)
print(disasterTrain) 

# clean Testing data
disasterTest = pd.read_csv('test.csv')# load File
disasterTest.drop(columns = ['keyword', 'location'], inplace=True)

textRecords_test = disasterTest['text']
cleanRecords_test = cleanText(textRecords_test)
cleanRecords_test = pd.Series(cleanRecords_test)
disasterTest['CleanedText'] = cleanRecords_test
disasterTest.drop(columns = 'text', inplace=True)
print(disasterTest)

# Vectorise Text(train) input!
import sklearn   # import countVectorizer
count_vectorizer = feature_extraction.text.CountVectorizer()

disasterVectors_train = count_vectorizer.fit_transform(disasterTrain["CleanedText"])
disasterVectors_test = count_vectorizer.transform(disasterTest["CleanedText"])

print(disasterVectors_train[0].todense().shape)
print(disasterVectors_train[0].todense())

print(disasterVectors_test[0].todense().shape)
print(disasterVectors_test[0].todense())

from sklearn.ensemble import GradientBoostingClassifier
learning_rates = [0.01,0.03,0.1,0.3,1]
depths = range(1,15)
result = []

for learning_rate in learning_rates:
    for depth in depths:
        for_list = []
        gbrt = GradientBoostingClassifier(random_state=0,learning_rate=learning_rate,max_depth=depth)
        gbrt.fit(disasterVectors_train,disasterTrain.target)
        scores = model_selection.cross_val_score(gbrt, disasterVectors_train,disasterTrain.target, cv=3)
        for_list = [learning_rate,depth,scores,round(gbrt.score(disasterVectors_train,disasterTrain.target),3)]
        result.append(for_list)

for i in result:
    print(i,'\n')

sample_submission.head()
# save the dataframe locally
sample_submission.to_csv (r'C:\Users\Akshay Desai\Desktop\All Projects\Real or Not\sample_submission_gbrt.csv', index = False, header=True)