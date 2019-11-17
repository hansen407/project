#!/usr/bin/env python
# coding: utf-8

# # WORD2VEC

# In[1]:


import gensim
import os
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.tokenize import sent_tokenize, word_tokenize


# In[2]:


sentences = open("berita.txt") 
s = sentences.read()


# In[3]:


f = s.replace("\n", " ") 
  
data = [] 


# In[4]:


for i in sent_tokenize(f): 
    temp = [] 
      
    # tokenize the sentence into words 
    for j in word_tokenize(i): 
        temp.append(j.lower()) 
  
    data.append(temp)


# In[5]:


model5 = gensim.models.Word2Vec(data, size=150, window=10, workers=10, sg=1, min_count = 1)

print(model5)


# In[6]:


words = list(model5.wv.vocab)
print(words)


# In[7]:


model5.wv.most_similar("hari")


# In[8]:


print(model5.similarity('hari', 'apa')) 


# In[9]:


model5.train(data,total_examples=len(data),epochs=25)


# In[10]:


model5.save('training5.model')

model5.save('training5.h5')

model5.save('training5.json')


# In[13]:


#model5.train(model5,total_examples=len(data), epoch = 5)

#or try this one

#model5.train(words, total_examples=len(data), epoch=5)


# In[11]:


w1 = "hari"
model5.wv.most_similar (positive=w1)


# In[20]:


print(model5.similarity('hari', 'rabu')) 


# In[21]:


print(model5['korupsi'], model5['korupsi'].shape)


# In[42]:


print(model5.wv['saya'])


# In[22]:


#importing libraries
import spacy
from spacy.vocab import Vocab
import numpy
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Embedding
from keras.models import load_model
import pickle


# In[23]:


data = open('1525132860000.detik-news.d0d7c1aa99.json').read()[:100000]
data


# In[24]:


#function for preparing text data into sequences for training 
def data_sequencing(data):   
    # integer encode sequences of words
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([data])
    with open('tokenizer.pkl', 'wb') as f: # Save the tokeniser by pickling it
        pickle.dump(tokenizer, f)

    encoded = tokenizer.texts_to_sequences([data])[0]
    # retrieve vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)
    
    # create line-based sequences
    sequences = list()
    rev_sequences = list()
    for line in data.split('.'):
        encoded = tokenizer.texts_to_sequences([line])[0]
        rev_encoded = encoded[::-1]
        for i in range(1, len(encoded)):
            sequence = encoded[:i+1]
            rev_sequence = rev_encoded[:i+1]
            sequences.append(sequence)
            rev_sequences.append(rev_sequence)
    print('Total Sequences: %d' % len(sequences))
    
    
    #find max sequence length 
    max_length = max([len(seq) for seq in sequences])
    with open('max_length.pkl', 'wb') as f: # Save max_length by pickling it
        pickle.dump(max_length, f)
    print('Max Sequence Length: %d' % max_length)

    # pad sequences and create the forward sequence
    sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
    # split into input and output elements
    sequences = array(sequences)
    X, y = sequences[:,:-1],sequences[:,-1]
    
    #pad sequences and create the reverse sequencing
    rev_sequences = pad_sequences(rev_sequences, maxlen=max_length, padding='pre')
    # split into input and output elements
    rev_sequences = array(rev_sequences)
    rev_X, rev_y = rev_sequences[:,:-1],rev_sequences[:,-1]

    return X,y,rev_X,rev_y,max_length,vocab_size


# In[25]:


#returning forward and reverse sequences along with max sequence 
#length from the data 

X,y,rev_X,rev_y,max_length,vocab_size = data_sequencing(data)


# In[26]:


# define forward sequence model
model = Sequential()
model.add(Embedding(vocab_size,100, input_length=max_length-1))
#model.add(LSTM(100))
model.add(Bidirectional(LSTM(100)))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())


# In[27]:


# define reverse model
rev_model = Sequential()
rev_model.add(Embedding(vocab_size, 100, input_length=max_length-1))
#rev_model.add(LSTM(100))
rev_model.add(Bidirectional(LSTM(100)))
rev_model.add(Dense(vocab_size, activation='softmax'))
print(rev_model.summary())


# In[28]:


# compile forward sequence network
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(X, y,batch_size=100, epochs=100, verbose=2)
# save the model to file
model.save('model.h5')


# In[29]:


# compile reverse sequence network
rev_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
rev_model.fit(rev_X, rev_y,batch_size=100, epochs=100, verbose=2)
# save the model to file
rev_model.save('rev_model.h5')


# In[30]:


# generate a sequence using a language model
def generate_seq(model, tokenizer, max_length, seed_text):
    if seed_text == "":
        return ""
    else:
        in_text = seed_text
        n_words = 1
        n_preds = 5 #number of words to predict for the seed text
        pred_words = ""
        # generate a fixed number of words
        for _ in range(n_words):
            # encode the text as integer
            encoded = tokenizer.texts_to_sequences([in_text])[0]
            # pre-pad sequences to a fixed length
            encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
            # predict probabilities for each word
            proba = model.predict(encoded, verbose=0).flatten()
            #take the n_preds highest probability classes 
            yhat = numpy.argsort(-proba)[:n_preds] 
            # map predicted words index to word
            out_word = ''

            for _ in range(n_preds):
                for word, index in tokenizer.word_index.items():
                    if index == yhat[_] and word not in stopwords:
                        out_word = word
                        pred_words += ' ' + out_word
                        #print(out_word)
                        break


        return pred_words


# In[43]:


# load the model
model = load_model('model.h5')
rev_model = load_model('rev_model.h5')
nlp = KeyedVectors.load('training5.model')

#load tokeniser and max_length
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
    
with open('max_length.pkl', 'rb') as f:
    max_length = pickle.load(f)
    
#loading stopwords to improve relevant word predictions    
#stopwords= open('stopwords').read().split()

#load spacy GloVe Model


# In[37]:


#Find and set embeddings for OOV words
def set_embedding_for_oov(doc):
    #checking for oov words and adding embedding
    for token in doc:
        if token.is_oov == True:
            before_text = doc[:token.i].text
            after_text = str(array(doc)[:token.i:-1]).replace('[','').replace(']','')

            pred_before = generate_seq(model, tokenizer, max_length-1, before_text).split()
            pred_after = generate_seq(rev_model, tokenizer, max_length-1, after_text).split()
            
            embedding = numpy.zeros((300,))

            i=len(before_text)
            print('Words predicted from forward sequence model:')
            for word in pred_before:
                print(word)
                embedding += i*nlp.vocab.get_vector(word)
                i= i*.5
            i=len(after_text)
            print('Words predicted from reverse sequence model:')
            for word in pred_after:
                print(word)
                embedding += i*nlp.vocab.get_vector(word)
                i= i*.5
            nlp.vocab.set_vector(token.text, embedding)
            print(token.text,nlp.vocab.get_vector(token.text))           


# In[47]:


doc = nlp('upaya pencarian dihadang awan tebal ')
set_embedding_for_oov(doc)


# In[50]:


print(nlp['saya', 'sedang', 'makan', 'nasi'])


# In[61]:


test_word_vector = array([-1.62694916e-01, -3.94468963e-01,  5.55681765e-01,  5.65985143e-02,
  -4.68945917e-04, -3.00415128e-01,  1.92553520e-01,  5.12435377e-01,
   1.52510870e-02, -3.49852383e-01,  1.17014743e-01,  4.03364241e-01,
  -4.18998122e-01, -5.26227474e-01,  3.27333510e-01, -3.03795725e-01,
   3.96258056e-01,  1.88540131e-01, -2.70056307e-01, -2.30784833e-01,
   2.88182110e-01, -2.92525440e-01,  4.56014574e-01,  8.89170244e-02,
   4.51279879e-01, -7.67208338e-01,  3.23082596e-01, -2.08701435e-02,
  -2.78338380e-02, -1.63007855e-01,  3.25572044e-01, -5.04837871e-01,
  -1.19281903e-01,  8.98872614e-02,  5.22464737e-02,  2.37989426e-01,
  -1.18570760e-01, -2.93719888e-01,  3.37260783e-01,  1.46512896e-01,
   5.45875907e-01,  4.25006509e-01, -2.15530396e-01,  3.91755670e-01,
   4.13839161e-01, -2.95187593e-01,  3.39853108e-01,  4.06097531e-01,
   1.30156085e-01, -1.56865224e-01,  6.65572062e-02,  2.45107278e-01,
  -3.74509171e-02,  5.37032001e-02,  1.53138414e-01,  3.81106138e-01,
   8.99646431e-02,  3.40260714e-01, -1.19019471e-01, -2.44506210e-01,
  -3.44622463e-01, -5.25708258e-01, -5.95613010e-02, -2.46109571e-02,
   2.91679204e-01, -8.02707672e-02, -3.26028198e-01, -2.28712484e-01,
  -1.48456529e-01, -1.53582454e-01,  1.41336799e-01,  3.54012012e-01,
   1.69483066e-01, -3.54806967e-02, -6.11210585e-01, -3.30946892e-01,
   5.62143372e-03,  3.44962329e-01,  1.05015025e-01, -1.12083606e-01,
  -7.56384075e-01,  3.58438164e-01,  5.55390567e-02,  8.11216310e-02,
  -7.37215206e-02,  5.45039326e-02,  4.06679884e-03,  2.98570782e-01,
  -3.92199695e-01, -1.69417471e-01, -2.89781749e-01,  3.98627251e-01,
   3.71541202e-01, -2.40003675e-01, -1.23068906e-01, -4.71732765e-01,
  -5.63765224e-03,  3.63860250e-01,  1.15735590e+00, -3.09262693e-01,
   7.25577548e-02, -1.46407828e-01, -6.60637558e-01, -1.28581822e-01,
  -5.32191336e-01,  3.30290318e-01, -1.02478035e-01,  2.49489173e-01,
  -1.09088644e-01,  1.25453487e-01,  2.31981561e-01, 3.32219005e-02,
   1.14234956e-02, -1.03443168e-01, -5.21015465e-01,  2.62629181e-01,
   1.78104728e-01,  1.88880131e-01, -1.44195870e-01,  3.70161422e-02,
  -4.63218391e-01, -6.14292502e-01, -2.21147478e-01,  4.06262189e-01,
   3.37421954e-01,  3.32743853e-01, -9.56421793e-02, -2.37437770e-01,
   3.20782542e-01, -1.11977786e-01,  5.51898777e-01,  2.88444310e-01,
   5.96565127e-01,  1.13103427e-01,  1.02439024e-01, -2.01034725e-01,
  -1.34422332e-01, -4.22162175e-01, -3.35944793e-03, -4.35981840e-01,
  -2.81984597e-01,  5.03499985e-01, -3.50894153e-01,  2.69655380e-02,
   2.96180964e-01,  1.36547029e-01,  3.85746866e-01, -1.79692477e-01,
   3.27225804e-01, -1.21459901e-01])


# In[62]:


nlp.most_similar(positive=[test_word_vector])


# In[ ]:




