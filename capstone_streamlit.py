import streamlit as st
import pandas as pd 
import numpy as np 
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences


#bring in function for preprocessing
## make each single letter amino acid it's own "word" for tokenizing
def make_words(sequence):
    
    """
    argument: amino acid sequence in single letter code/abbreviation
    output: input amino acid sequence with spaces between each single letter code
    """
    
    
    #instantiate empty string to concat amino acids to
    space_seq = ''
    
    #for loop to loop over each amino acid in the sequence and add it into the 
    #string with a space afterwards
    for char in sequence:
        space_seq += char
        space_seq += " "
    #remove the space at end of the new string
    return space_seq[:-1] 


#header for the app
st.header('Protein Function Predictor')


#write some text to the top of the app
st.write('This app will take in a single-letter amino acid sequence and return both a predicted function class and the probability the protein sequence aligns with one of the 8 classes.')
st.write('The app is not comprehensive, as it only covers the following GO molecular function classes:')

#create a function to be cached for preprocessing the model info
@st.cache(suppress_st_warning=True)
def preprocess():
    #transform the string to numerical representation
    #read in the data to get the tokenized words
    protein = pd.read_pickle('data/compressed-class-separated.pkl', compression = 'gzip')
    #preprocess each sequence in X
    protein['Sequence'] = protein['Sequence'].apply(lambda x: make_words(x))

    #teach the tokenizer the 20 amino acid "words"
    X = protein['Sequence']
    y = protein['class']
    X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, random_state = 18) 
    tokenizer = Tokenizer(20)
    tokenizer.fit_on_texts(X_train)

    return tokenizer

#create dictionary of classes
class_dict = {
    'Class 1': 'rRNA Binding [GO:0019843], structural component of the ribosome [GO:0003735]',
    'Class 2': 'DNA Binding [GO:0003677]',
    'Class 3': 'ATP Binding [GO:0005524]',
    'Class 4': 'Hormone Activity [GO:0005179]',
    'Class 5': 'GTPase Activity [GO:0003924]',
    'Class 6': 'NADH Dehydrogenase (Ubiquinone) Activity [GO:0008137] & (Quinone) Activity [GO:0050136]',
    'Class 7': 'Oxidoreductase Activity [GO:0016491]',
    'Class 8': 'Toxin Activity [GO:0090729]'
}

#create table of the different classes
df = pd.DataFrame(class_dict, index = ['GO Molecular Function']).T


#add dataframe to the streamlit app
st.dataframe(df)


#add some whitespace to the app
st.text('')
st.text('')


#insert the prediction section
st.header('Predict Function Class from RNN Model')


#input should be in single-letter amino acid code
#allow input of string in text box
    #upper case all letters
seq_input = st.text_area(label = 'Input Single-Letter Amino Acid Sequence', max_chars=None, key = 'sequence_to_predict').upper()
#preprocess the sequence added
#remove all white space from the sequence input
seq_input = seq_input.replace(" ","")

if seq_input.isalpha():

    #add in spaces between each 1-letter amino acid to make them "words"
    transformed_input = make_words(seq_input)

    #call preprocess
    tokenizer = preprocess()

    #pad the sequence
    #num_input = pad_sequences
    #make the user input into a number
    num_input = tokenizer.texts_to_sequences([transformed_input])
    num_input = pad_sequences(num_input, maxlen = 18_000)

    #load in the model
    loaded_model = keras.models.load_model('rnn_model_sampled_2.h5')

    #pass the user input sequence to the model
    preds = loaded_model.predict(num_input)
    #set up indexes for pred dataframe
    i = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8']

    #align the predictions with each class
    preds_df = pd.DataFrame(preds, columns= i, index=['Prediction Probabilities'])

    #print the predictions to the streamlit app
    st.table(preds_df)

    #write out the prediction as the highest probability from those returned
    class_pred = np.argmax(preds, axis = 1)[0]
    class_key = list(class_dict.values())[class_pred]
    class_pred += 1

    #write out the models predictions
    st.header('Model Prediction:')
    st.write(f'Highest Predicted Class based on Probability: {class_pred}.')
    st.write(f'Class {class_pred} = {class_key}')



else:
    st.write('The 1-letter amino acid sequence can only contain letters.')

