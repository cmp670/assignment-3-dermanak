
# coding: utf-8

import random
import math

import nltk
import dynet as dy
import numpy as np

from scipy.sparse import dok_matrix


# Defining dynet functions for forward network
# Include two methods, fit and predict
# The method fit: trains the network and the method predict: forward propagation
class HiddenLayeredNetwork:

    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.001):
        
        self.mdl = dy.Model() # Create model
        self.sgd = dy.MomentumSGDTrainer(self.mdl, learning_rate=learning_rate)
        
        # Constructing weights and biasees
        self.W1 = self.mdl.add_parameters((hidden_dim, input_dim))
        self.hbias = self.mdl.add_parameters((hidden_dim, ))
        self.W2 = self.mdl.add_parameters((output_dim, hidden_dim))

    # Training the network
    def partial_fit(self, X, y):
        
        dy.renew_cg()
        losses = []

        # Calculation of loss for each input
        for inp, out in zip(X, y):
            inp = dy.inputVector(inp)

            #Relu function
            h = dy.rectify(self.W1 * inp + self.hbias)
            #calculating output on the last layer
            logits = self.W2 * h

            #Applying softmax and calculating loss
            loss = dy.pickneglogsoftmax(logits, out)
            losses.append(loss)
        mbloss = dy.esum(losses) / X.shape[0]
        mbloss.backward()
        self.sgd.update()
        
        eloss = mbloss.scalar_value()

    # Predict method calculates probability of P(Y=y | X) for all y values
    def predict_proba(self, x):
        x = dy.inputVector(x)
        h = dy.rectify(self.W1 * x + self.hbias)
        logits = self.W2 * h
    
        # Converting outputs to probabilities by using softmax function
        temp = np.exp(logits.npvalue())
        prob_lst = temp / np.sum(temp)
        return prob_lst


filename = 'trumpspeeches.txt'


# Reading data
with open(filename, encoding='utf-8') as file:
    data = file.read()

data = data.lower()

# Converting words into vectors
word2index = dict()

word2index['$START$'] = 0
word2index['$END$'] = 1

# Returning the index of the word that exist, otherwise give a new index.
def get_index(word):
    
    if word not in word2index:
        word2index[word] = len(word2index)
    
    return word2index[word]


# Tokenizing

sent_lst = nltk.sent_tokenize(data)
word_set = set()

# Counting each word occurence and each unique word occurence
bigram_count = 0
for i in range(len(sent_lst)):
    sent_lst[i] = nltk.word_tokenize(sent_lst[i])
    
    for word in sent_lst[i]:
        word_set.add(word)
    
    bigram_count += len(sent_lst[i]) + 1
    
word_count = len(word_set) + 2

# Encode

# Defining one hot vectors
X = dok_matrix((bigram_count, word_count), dtype=int)
y = [-1 for _ in range(bigram_count)]

index_lst = list(range(bigram_count))
random.shuffle(index_lst)

counter = 0
for sent in sent_lst:
    pre_word = 0 # 0 corresponds to $START$ TOKEN
    for word in sent:
        current_word = get_index(word)
        index = index_lst[counter]
        counter += 1
        X[index, pre_word] = 1
        y[index] = current_word
    
        pre_word = current_word
    current_word = 1 # 1 corresponds to $END$ TOKEN
    
    index = index_lst[counter]
    counter += 1
    X[index, pre_word] = 1
    y[index] = current_word

# Converting data to a sparse matrix structure since the data is very sparse
X = X.tocsr()

# Creating model

mdl = HiddenLayeredNetwork(input_dim=word_count, hidden_dim=1024, output_dim=word_count, learning_rate=0.5)


# Network training part

iteration_count = 2000
batch = 15
batch_count = math.ceil(bigram_count / batch)

for i in range(iteration_count):
    index = i % batch_count
    start = index * batch
    
    if i == batch_count - 1:
        end = bigram_count
    else:
        end = (index + 1) * batch
    
    inp = np.array(X[start:end].todense())
    mdl.partial_fit(inp, y[start:end])


# Constructing reverse of word2index to get the output as a word
index2word = [None for _ in range(word_count)]
for word in word2index:
    index2word[word2index[word]] = word

#Generating sentences which have max 50 words
def generate_sent(max_length=50):
    pre_word = 0
    word_lst = [] # stores all generated words
    
    current_word = None
    while(current_word != 1):
        x_predict = np.zeros(word_count)
        x_predict[pre_word] = 1

        # Returning occurence probabilities of all words
        prob_lst = mdl.predict_proba(x_predict)

        # Chosing a random word to start
        value = random.random()

        total = 0
        for i in range(word_count):
            total += prob_lst[i]
            if value < total:
                current_word = i
                break
                
        if current_word == 1:
            break

        word = index2word[current_word]
        word_lst.append(word)

        if len(word_lst) == max_length:
            return word_lst
        
        pre_word = current_word
    
    return word_lst

# Generate 5 random sentences
for i in range(5):
    word_lst = generate_sent()
    print(*word_lst)

