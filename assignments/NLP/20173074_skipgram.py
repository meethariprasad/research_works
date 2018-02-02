
# coding: utf-8

# In[1]:


print ("Based on : https://arxiv.org/abs/1411.2738 for forward and backpropogation derivations of Word2Vec CBOW & Skipgram.")
#numpy version '1.13.1'


# In[ ]:


print ("######################")
print ("\n Make sure you have internet connection (2-5 Mbps),Tensorflow, math, numpy-1.13.1, NLTK (with data),random, os,six.moves libraries in Python 3.6.3 exists.\n")
print ("######################")
target_list = input("Enter comma seperated words for which you want to predict the nearest words: ")


# In[ ]:


print ("Tokenizing inputs in to words")
from nltk.tokenize import word_tokenize
target_tokens = word_tokenize(target_list)
print ("Convert to lower case")
target_tokens = [w.lower() for w in target_tokens]
# remove punctuation from each word
import string
table = str.maketrans('', '', string.punctuation)
target_stripped = [w.translate(table) for w in target_tokens]
print ("remove punctuation from each word, if any")
# remove remaining tokens that are not alphabetic
target_words = [word for word in target_stripped if word.isalpha()]
print ("remove remaining tokens that are not alphabetic, if any: ",target_words)
#Converting the target_words to tuple as that is what the further functions expect.
target_words=tuple(target_words)


# In[3]:


print ("######################")
download_embed = input("To download pre-trained embeddings(For faster predictions), enter 1, else 0 for training 100000 iterations over data::")
print ("######################")


# # Download embeddings

# In[4]:


import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination) 

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


# In[ ]:


#Download the file if not yet downloaded.
   #https://drive.google.com/file/d/1rI79RgJx2ZajOGtYsw-kxCjrr2VUYH1e/view?usp=sharing
   #id=1rI79RgJx2ZajOGtYsw-kxCjrr2VUYH1e
import os.path
if((os.path.isfile("corpus.txt")!=True)):
   print ("Downloading corpus.txt as it is not in path.")
   id="1rI79RgJx2ZajOGtYsw-kxCjrr2VUYH1e"
   destination="corpus.txt"
   download_file_from_google_drive(id,destination)


# In[5]:


#Download embeddings.
#https://drive.google.com/file/d/1LiBLVn1Q3-a2GndvnA8-vRlTR9kNBOdE/view?usp=sharing
import os.path
if(int(download_embed)==1):
    id="1LiBLVn1Q3-a2GndvnA8-vRlTR9kNBOdE"
    print ("Downloading skipgram_embed.npy as per user choice.")
    destination="skipgram_embed.npy"
    download_file_from_google_drive(id,destination)
elif ((os.path.isfile("skip_embed.py")==True) & (int(download_embed)!=1) ):
    os.remove("skip_embed.py")
else:
    print ("Unknown error")


# In[6]:


import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
#We are not using tensor flow implementation as backpropogation is implemented using numpy itself.
import tensorflow as tf
from keras.utils import np_utils


# In[7]:


def corpus2word(text):
    from keras.preprocessing.text import text_to_word_sequence
    # define the document
    # tokenize the document
    words = text_to_word_sequence(text)
    return words


# In[8]:


import os
import random
from six.moves import range
import numpy as np
import collections

print ("######################")
if ((os.path.isfile("dictionary.npy")==True) & (os.path.isfile("reverse_dictionary.npy")==True) & (os.path.isfile("words.npy")==True)):
    print("The corpus will not be processed for pre cleansing as it has been already processed")
else:
    print("Reading and cleaning corpus to train")
    file = open('corpus.txt', 'r')
    text = file.read()
    file.close()
    #NLTK for text processing
    import nltk
    print("Removing non english words.")
    eng_words = set(nltk.corpus.words.words())

    text_english=" ".join(w for w in nltk.wordpunct_tokenize(text)              if w.lower() in eng_words or not w.isalpha())
    #Text Tokenization
    print("Split into words")
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text_english)
    print("Convert to lower case")
    tokens = [w.lower() for w in tokens]
    print("Remove punctuation from each word")
    import string
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    print("Remove remaining tokens that are not alphabetic")
    words = [word for word in stripped if word.isalpha()]
    print("Filter out stop words")
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    print("Removed Single Letter words")
    single_letters=('b','c','d','e','f','g','h','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z')
    words=[w for w in words if not w in single_letters]
    print("Total words in corpus")
    print (len(words))
    print("Total unique words")
    print (len(set(words)))
    np.save("words.npy",words)


# In[9]:


import collections

def build_dataset(words):
    count = [['UNK', -1]]
    vocabulary_size = len(set(words))-1
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
    for word in words:
        if word in dictionary:
          index = dictionary[word]
        else:
          index = 0  # dictionary['UNK']
          unk_count = unk_count + 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
    vocabulary = len(set(dictionary))-1
    data_in_list=list()
    data_in_list.append(data)
    return data_in_list, count, dictionary, reverse_dictionary,vocabulary


# In[10]:


print ("######################Creating Dictionary, Reverse Dictionary, Sequencing the Integers#############")
if ((os.path.isfile("dictionary.npy")==True) & (os.path.isfile("reverse_dictionary.npy")==True) & (os.path.isfile("data.npy")==True)):
    print("The Dictionary will not be processed for creating dictionary as it has been already processed.")
else:
    data, count, dictionary, reverse_dictionary,V = build_dataset(words)
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10])
    import numpy as np
    # Save
    np.save('dictionary.npy', dictionary)
    np.save('reverse_dictionary.npy', reverse_dictionary)
    np.save('data.npy', data)


# In[11]:


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical
def corpus2io(corpus_tokenized, V, window_size):
    """Converts corpus text into context and center words
    # Arguments
        corpus_tokenized: corpus text
        window_size: size of context window
    # Returns
        context and center words (arrays)
    """
    for words in corpus_tokenized:
        L = len(words)
        for index, word in enumerate(words):
            contexts = []
            labels = []
            s = index - window_size
            e = index + window_size + 1
            contexts.append([words[i]-1 for i in range(s, e) if 0 <= i < L and i != index])
            labels.append(word-1)
            x = np_utils.to_categorical(contexts, V)
            y = np_utils.to_categorical(labels, V)
            yield (x, y.ravel())


# In[12]:


def softmax(x):
    """Calculate softmax based probability for given input vector
    # Arguments
        x: numpy array/list
    # Returns
        softmax of input array
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


# In[13]:


def initialize(V, N):
    """
    Initialize the weights of the neural network.
    :param V: size of the vocabulary
    :param N: size of the hidden layer
    :return: weights W1, W2
    """
    np.random.seed(100)
    W1 = np.random.rand(V, N)
    W2 = np.random.rand(N, V)

    return W1, W2


# In[14]:


def skipgram(context, x, W1, W2, loss):
    """
    Implementation of Skip-Gram Word2Vec model
    :param context: all the context words (these represent the labels)
    :param label: the center word (this represents the input)
    :param W1: weights from the input to the hidden layer
    :param W2: weights from the hidden to the output layer
    :param loss: float that represents the current value of the loss function
    :return: updated weights and loss
    """
    h = np.dot(W1.T, x)
    u = np.dot(W2.T, h)
    y_pred = softmax(u)

    e = np.array([-label + y_pred.T for label in context])
    dW2 = np.outer(h, np.sum(e.reshape(e.shape[1],e.shape[2]), axis=0))
    dW1 = np.outer(x, np.dot(W2, np.sum(e.reshape(e.shape[1],e.shape[2]), axis=0).T))
    new_W1 = W1 - eta * dW1
    new_W2 = W2 - eta * dW2
    loss += -np.sum([u[label == 1] for label in context.reshape(context.shape[1],context.shape[2])])
    return new_W1, new_W2, loss


# In[15]:


words=np.load("words.npy")
corpus_tokenized,_,_,_,V = build_dataset(words)
N=150
W1, W2 = initialize(V, N)
np.random.seed(100)
n_epochs=1
window_size=2
eta = 0.1
loss_vs_epoch = []
if ((os.path.isfile("skipgram_embed.npy")!=True) & (int(download_embed)!=1)):
    for e in range(n_epochs):
        loss = 0
        for i, (label, center) in enumerate(corpus2io(corpus_tokenized, V, window_size)):
            W1, W2, loss = skipgram(label, center, W1, W2, loss)
            loss_vs_epoch.append(loss)
    np.save("skipgram_embed.npy", W1)


# In[16]:


#Predict function
#Creating predict function
#Input: list of words, final_embeddings,reverse_dictionary,top_nearest_words_needed
#Output: The nearest 10 words in sorted order.
def predicted_words(target_words,final_embeddings,dictionary,reverse_dictionary,top_nearest_words_needed):

    #Handling the single input and multiple inputs, both.
    target_list=[]
    if (type(target_words)==tuple):
        for i in range(0,len(target_words)):
            target_list.append(target_words[i])
    else:
        target_list.append(target_words)
    
    #Take word by word and predict list of nearest words.
    for i in range(0,len(target_list)):
        #Cleaning target word.
        target=target_list[i]
        #search for index in dictionary. If not found assign it to unknown word.
        #The results will be obviously not accurate, but will not give annoying not found error.
        #Approach is debatable! 
        #But I am following approach in google keypad, when I enter non existing word, it will still give some junk predictions.
        if(dictionary.get(target)==None):
            target_embedding=final_embeddings[0,:]
        else:
            target_embedding=final_embeddings[int(dictionary.get(target))-1,:]
        
        if(dictionary.get(target)!=None):
            
            #cosine_similarity
            cosine_similarity=np.matmul(target_embedding,np.transpose(final_embeddings[0:final_embeddings.shape[0],:]))
        
            #I am building a array with index in one column and cosine similarity in another column.
            word_index=np.zeros((cosine_similarity.shape[0],1), dtype=int)
        
            for j in range(0,cosine_similarity.shape[0]):
                word_index[j]=j+1
        
            #Reshaping cosine_similarity to match with word index array.
            cosine_similarity_reshape=cosine_similarity.reshape(cosine_similarity.shape[0],1)
        
            #Appending & Sorting.
            target_sim_array=np.append(word_index,cosine_similarity_reshape,axis=1)
            target_sim_array_sorted=target_sim_array[target_sim_array[:, 1].argsort()]
        
            #Taking top nearest word index except word itself (0 th position)
            top_word_index=target_sim_array_sorted[:,0][-(top_nearest_words_needed+1):-1:]
        
            #Top word predictions using reverse array and reverse dictionary
            predicted_words=list()
            for k in reversed(top_word_index):
                predicted_words.append(reverse_dictionary[k])
            
            print ("Nearest words for word: %s" %target, ": ordered by nearest word predicted first is %s" %predicted_words)
        else:
                      
            print ("Unknown word yet in dictionary, Add sentences with this word in corpus and train again to predict: %s" %target)  


# In[17]:


import numpy as np
# Load embeddings, dictionary and reverse dictionary saved earlier in files.
# Advantage is you can reduce run time, if these files doesn't exist.
final_embeddings = np.load("skipgram_embed.npy")
dictionary = np.load('dictionary.npy').item()
reverse_dictionary = np.load('reverse_dictionary.npy').item()


# In[21]:


target_words=target_words
top_nearest_words_needed=10


# In[22]:


print ("###########################")
predicted_words(target_words,final_embeddings,dictionary,reverse_dictionary,top_nearest_words_needed)

