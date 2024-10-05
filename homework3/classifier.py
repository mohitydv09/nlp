import argparse

import nltk
from nltk.util import ngrams
from nltk import FreqDist
from nltk.lm import MLE
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.lm.preprocessing import pad_both_ends, flatten, padded_everygram_pipeline

import numpy as np

## I am using NLTK tokenizer to tokenize the text.
nltk.download('punkt')

def split_data(data:list[list[str]], test_size:float=0.1)->tuple[list[list[str]], list[list[str]]]:
    ## Split the data into training and testing data.
    training_data = [text[:int(len(text)*(1-test_size))] for text in data]
    testing_data = [text[int(len(text)*(1-test_size)):] for text in data]

    return training_data, testing_data

def preprocess_data(N:int, data:list[list[str]])->list:

    ## Tokenize the text
    data = [[word_tokenize(text) for text in author] for author in data]

    processed_data = [padded_everygram_pipeline(N, text) for text in data]
    return processed_data

def argument_parser():
    parser = argparse.ArgumentParser(description='Authorship classifier')
    parser.add_argument('authorlist', type=str, help='Path to file containing list of authors')
    parser.add_argument('-approach', 
                        type=str, 
                        choices=['generative','descriminative'], 
                        required=False,
                        help='Choose the approach to classify the author: Options: generative, discriminative')
    parser.add_argument('-test',
                        type=str,
                        required=False,
                        help='Provide the path of file to test.')
    args = parser.parse_args()

    authorlist = [line.strip() for line in  open(args.authorlist, "r").readlines()]

    return authorlist, args.approach, args.test

if __name__ == '__main__':
    authorlist, approach, test = argument_parser()

    ## Load the data and tokenize into sentences.
    data = [open("./ngram_authorship_train/"+filename, "r").read().lower() for filename in authorlist]
    data = [sent_tokenize(text) for text in data]

    ## Set N for n-gram
    N = 2

    if test:
        print("Test file provided, train with whole data")
        training_data = preprocess_data(N, data) ## Data is a list[list]

        ## Load and process the test data
        print("Implement correct test data loading.")
        testing_data = open(test, "r").read().lower()
        testing_data = [sent_tokenize(testing_data)] ### Converted to list to standardize the data format.
        testing_data = preprocess_data(N, testing_data)
    else:
        print("Test file not provided, creating data with 90% train and 10% test")
        training_data, testing_data = split_data(data)

        ## Process the data.
        training_data = preprocess_data(N, training_data)
        testing_data = preprocess_data(N, testing_data)

    if approach == 'generative':
        print('Generative approach selected')

        ## Train the model
        ngram_models = [MLE(N) for _ in range(len(authorlist))]

        ## Train the models.
        for model, data in zip(ngram_models, training_data):
            model.fit(data[0], data[1])
        
        testing_data_ngrams = [item[0] for item in testing_data]

        if test:
            print("Do something with the test data")

        else:
            results = np.zeros((len(authorlist),len(testing_data_ngrams)))
            for i,text_data in enumerate(testing_data_ngrams):
                # size_text_data = len(text_data)
                for sentence in text_data:
                    sentence = list(sentence)

                    predictions = [model.perplexity(sentence) for model in ngram_models]
                    if np.min(predictions) == np.inf:
                        continue
                    prediction = np.argmin(predictions)
                    # print("Prediction: ", prediction)
                    results[i,prediction] += 1
                # results[i] = results[i]/size_text_data
        print(results)
        print(np.sum(results, axis=1))
        results = results/np.sum(results, axis=1, keepdims=True)
        print(results)