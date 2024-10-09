import argparse
import numpy as np

from nltk.lm import MLE, Laplace, KneserNeyInterpolated, StupidBackoff, Vocabulary, AbsoluteDiscountingInterpolated
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm.counter import NgramCounter

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from tabulate import tabulate

import wandb

wandb.init(mode='disabled')
## I am using NLTK tokenizer to tokenize the text.
# nltk.download('punkt')

def split_data(data, test_size:float=0.1):
    ## Split the data into training and testing data.
    training_data = [text[:int(len(text)*(1-test_size))] for text in data]
    testing_data = [text[int(len(text)*(1-test_size)):] for text in data]

    return training_data, testing_data

def preprocess_data(N, data):

    ## Tokenize the text
    data = [[word_tokenize(text) for text in author] for author in data]

    processed_data = [padded_everygram_pipeline(N, text) for text in data]
    return processed_data

def argument_parser():
    parser = argparse.ArgumentParser(description='Authorship classifier')
    parser.add_argument('authorlist', type=str, help='Path to file containing list of authors')
    parser.add_argument('-approach', 
                        type=str, 
                        choices=['generative','discriminative'], 
                        required=True,
                        help='Choose the approach to classify the author: Options: generative, discriminative')
    parser.add_argument('-test',
                        type=str,
                        required=False,
                        help='Provide the path of file to test.')
    args = parser.parse_args()

    authorlist = [line.strip() for line in  open(args.authorlist, "r").readlines()]

    return authorlist, args.approach, args.test

def compute_metrics(p):  # Define the metrics to track
    predictions, labels = p
    preds = predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

def discriminative_model(authors, training_dataloader, testing_dataloader):  

    wandb.login()

    wandb.init(project="authorship_classifier", 
               config={
                   "num_epochs":10,
                   "batch_size":1024,
                   "learning_rate":1e-4,
               })

    num_authors = len(authors)
    id2label = {i:author.split('.')[0] for i,author in enumerate(authors)}
    label2id = {author.split('.')[0]:i for i,author in enumerate(authors)}

    training_args = TrainingArguments(
        output_dir="./results",
        run_name="authorship_classifier",
        num_train_epochs=wandb.config.num_epochs,
        per_device_train_batch_size=wandb.config.batch_size,
        per_device_eval_batch_size=wandb.config.batch_size,
        learning_rate=wandb.config.learning_rate,
        warmup_steps=1,
        weight_decay=0.01,
        logging_dir="./logs",
        report_to="wandb",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=5,
        save_steps=5,
        load_best_model_at_end=True
    )

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",
                                                               num_labels=num_authors,
                                                               id2label=id2label,
                                                               label2id=label2id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_dataloader,
        eval_dataset=testing_dataloader,
        compute_metrics=compute_metrics
    )

    trainer.train()

    trainer.save_model("./results")

    return model


class AuthorshipDataset(Dataset):
    def __init__(self, text, label):
        self.text = text
        self.label = label
    
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        return {"input_ids":self.text[idx]['input_ids'], 
                "label":self.label[idx],
                'attention_mask':self.text[idx]['attention_mask']}
    
class AuthorDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {'input_ids':self.data[idx]['input_ids'],
                'attention_mask':self.data[idx]['attention_mask'],
                'label':self.data[idx]['label']}

def test_discriminative_model(model, test_dataset):

    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics
    )

    ### Seperate the data author wise.
    author_datasets = [[] for _ in range(4)]
    for data_point in test_dataset:
        author_datasets[data_point["label"]].append(data_point)

    ## Convert the dataset to AuthorDataset.
    author_datasets = [AuthorDataset(dataset) for dataset in author_datasets]

    accuracies = []
    failure_cases = [[] for _ in range(4)]
    success_cases = [[] for _ in range(4)]
    for i,test_dataset in enumerate(author_datasets):
        predictions, label_ids, metrics = trainer.predict(test_dataset)
        predicted_labels = predictions.argmax(-1)
        for j,label in enumerate(predicted_labels):
            if label != i and len(failure_cases[i]) < 1:
                failure_cases[i].append(test_dataset[j])
            if label == i and len(success_cases[i]) < 1:
                success_cases[i].append(test_dataset[j])
        acc = accuracy_score(label_ids, predictions.argmax(-1))
        accuracies.append(acc)

    # # For printing the failure cases.
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", clean_up_tokenization_spaces=True)
    # print("Failure cases of Discriminative Model:")
    # for i, failure_case in enumerate(failure_cases):
    #     for item in failure_case:
    #         sentence = [word for word in item['input_ids'] if word != 0]
    #         sentence = tokenizer.decode(sentence)
    #         print("Failure Sentence:", sentence)

    # ## For printing the success cases.
    # for i, success_case in enumerate(success_cases):
    #     for item in success_case:
    #         sentence = [word for word in item['input_ids'] if word != 0]
    #         sentence = tokenizer.decode(sentence)
    #         print("Success Sentence:", sentence)

    print("Results on test set:")
    for i,author in enumerate(authorlist):
        author = author.split(".")[0]
        print(f"{author:<10}  {accuracies[i]*100:.2f}% correct")

def create_dataset_discriminative(data):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", clean_up_tokenization_spaces=True) ## Clean up tokenizeiotn added to supress warning.
    data_dict = {"text":[], "label":[]}
    for i, data_author in enumerate(data):
        data_dict["text"].extend([tokenizer(sentence, 
                                            padding="max_length",
                                            max_length=32, 
                                            truncation=True) for sentence in data_author])
        data_dict["label"].extend([i]*len(data_author))

    return AuthorshipDataset(data_dict["text"], data_dict["label"])


import torch
import torch.nn as nn
import torch.nn.functional as F
import re
#### BiGram model with Pytorch.

class BiGramModel(nn.Module):
    def __init__(self, vocab_size):
        super(BiGramModel, self).__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets = None):
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        ## idx is B,T as current context
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
    


if __name__ == '__main__':
    authorlist, approach, test = argument_parser()

    ## Load the data and tokenize into sentences.
    data = [open("./ngram_authorship_train/"+filename, "r").read().lower() for filename in authorlist]
    data = [sent_tokenize(text) for text in data]

    ## Set N for n-gram
    N = 2

    if approach == 'generative':
        ## Data Preprocessing.
        if test:
            print("Test file provided, train with whole data")
            training_data = preprocess_data(N, data) ## Data is a list[list]

            ## Load and process the test data. Assuming that each line is one sentence to predict.
            sentences = open(test, "r").readlines()
            testing_data = [[sentence.lower().strip()] for sentence in sentences]
            testing_data = preprocess_data(N, testing_data)
        else:
            training_data, testing_data = split_data(data)

            ## Process the data.
            training_data = preprocess_data(N, training_data)
            testing_data = preprocess_data(N, testing_data)

        ## You can StupidBackoff, AbsoluteDiscountingInterpolated, Laplace, MLE
        ngram_models = [Laplace(order=N, vocabulary=Vocabulary(unk_cutoff=1, unk_label='<UNK>')) for _ in range(len(authorlist))]

        ## Train the models.
        print("Training the models, this may take some time")
        training_data = [([list(ngram) for ngram in ngrams], list(vocab)) for ngrams, vocab in training_data]

        for model, data in zip(ngram_models, training_data):
            model.fit(data[0], data[1])

        # ## Getting the top-5 bigrams for each author.
        # from collections import Counter
        # counters = [Counter() for _ in range(len(authorlist))]
        # punctuation = [".", ",", "!", "?", ":", ";", "(", ")", "[", "]", "{", "}", "''", '""', "``", "’", "‘", "“", "”", "<s>", "</s>","'"]
        # for counter, (ngrams, vocab) in zip(counters,training_data):
        #     for sentence in ngrams:
        #         for ngram in sentence:
        #             if len(ngram) == 2 and ngram[0] not in punctuation and ngram[1] not in punctuation:
        #                 counter[ngram] += 1

        # table_data = []
        # for i, counter in enumerate(counters):
        #     top_bigrams = counter.most_common(5)
        #     for index, (bigram, count) in enumerate(top_bigrams):
        #         if index == 0:
        #             table_data.append([authorlist[i].split('.')[0].capitalize(), ' '.join(bigram), count])
        #         else:
        #             table_data.append(['', ' '.join(bigram), count])

        # # Print the table
        # print(tabulate(table_data, headers=["Author", "Bigram", "Count"], tablefmt="fancy_grid"))
        
        testing_data_ngrams = [item[0] for item in testing_data]
        if test:
            print("Predictions for each sentence in the test file:")
            for i,text_data in enumerate(testing_data_ngrams):
                for sentence in text_data:
                    sentence = list(sentence)
                    predictions = [model.perplexity(sentence) for model in ngram_models]
                    # if np.min(predictions) == np.inf:
                    #     print("No prediction, inf perplexity")
                    #     continue
                    prediction = np.argmin(predictions)
                    print(f"{authorlist[prediction].split('.')[0]}")

        else:
            results = np.zeros((len(authorlist),len(testing_data_ngrams)))
            failure_cases = [[] for _ in range(len(authorlist))]
            success_cases = [[] for _ in range(len(authorlist))]
            best_perplexity = [np.inf]*len(authorlist)
            best_sentence = [[]]*len(authorlist)
            for i,text_data in enumerate(testing_data_ngrams):
                for sentence in text_data:
                    sentence = list(sentence)
                    predictions = [model.perplexity(sentence) for model in ngram_models]
                    # for i, pred in enumerate(predictions):
                    #     if pred < best_perplexity[i]:
                    #         best_perplexity[i] = pred
                    #         best_sentence[i] = sentence
                    if np.min(predictions) == np.inf:
                        continue
                    prediction = np.argmin(predictions)
                    if prediction != i and len(failure_cases[i]) < 1:
                        failure_cases[i].append(sentence)
                    if prediction == i and len(success_cases[i]) < 1:
                        success_cases[i].append(sentence)
                    results[i,prediction] += 1
            results = results/np.sum(results, axis=1, keepdims=True)

            # ## Best sentence for each author.
            # ## Get monogram for the best sentence.
            # for i, sentence in enumerate(best_sentence):
            #     best_sentence[i] = [item[0] for item in sentence if len(item) == 1]

            # for i, author in enumerate(authorlist):
            #     author = author.split(".")[0]
            #     sentence = [word for word in best_sentence[i] if word != "</s>" and word != "<s>"]
            #     sentence = " ".join(sentence)
            #     print(f"{author:<10}  {sentence} Perplexity: {best_perplexity[i]}")

            # # Code to extract failure and success cases of generative model.
            # ##Get monograms from the failure cases.
            # for i, failure_case in enumerate(failure_cases):
            #     failure_cases[i] = [item[0] for item in failure_case[0] if len(item) == 1]
            #     failure_cases[i] = " ".join(failure_cases[i])

            # print("Failure cases of Generative Model:")
            # for failure_case, author in zip(failure_cases, authorlist):
            #     author = author.split(".")[0]
            #     print(f"{author:<10}  {failure_case}")

            # ##Get monograms from the success cases.
            # for i, success_case in enumerate(success_cases):
            #     success_cases[i] = [item[0] for item in success_case[0] if len(item) == 1]
            #     success_cases[i] = " ".join(success_cases[i])

            # print("Success cases of Generative Model:")
            # for success_case, author in zip(success_cases, authorlist):
            #     author = author.split(".")[0]
            #     print(f"{author:<10}  {success_case}")

            print("Results on dev set:")
            for i,author in enumerate(authorlist):
                author = author.split(".")[0]
                print(f"{author:<10}  {results[i,i]*100:.2f}% correct")

        #### Generating text from the models.
        # prompts = ["I want", "I went", "He said", "She said", "Go"]
        # for prompt in prompts:
        #     for i,model in enumerate(ngram_models):
        #         print(f"Generated text for {authorlist[i].split('.')[0]}")
        #         generated_text = model.generate(20, text_seed=prompt.split())
        #         filtered_text = " ".join([word for word in generated_text if word != "</s>" and word != "<s>"])
        #         ## Perplexity of the generated text.
        #         perplexity = model.perplexity(generated_text)
        #         print(f"{prompt} : {filtered_text}")
        #         print(f"Perplexity: {perplexity}")

    elif approach == 'discriminative':
        print("Descriminative approach selected.")

        ## Data Preprocessing for discriminative model.
        if test:
            print("Test file provided, train with whole data")
            training_data = data
            testing_data = open(test, "r").readlines()
            testing_data = [[sentence.lower().strip()] for sentence in testing_data]
        else:
            training_data, testing_data = split_data(data)
        training_dataset = create_dataset_discriminative(training_data)
        testing_dataset = create_dataset_discriminative(testing_data)

        # Train the model.
        model = discriminative_model(authorlist, training_dataset, testing_dataset)

        ## Load the already trained model.
        # model = AutoModelForSequenceClassification.from_pretrained("./results")

        ## Test the model.
        if test:
            print("Predictions for each sentence in the test file:")
            for i, text_data in enumerate(testing_dataset):
                predictions = model(input_ids=torch.tensor(text_data["input_ids"]).unsqueeze(0), 
                                    attention_mask=torch.tensor(text_data["attention_mask"]).unsqueeze(0))
                predictions = predictions.logits.argmax(-1)
                print(f"{authorlist[predictions].split('.')[0]}")
        else:
            test_discriminative_model(model, testing_dataset)

    test_bigram_pytorch = False
    if test_bigram_pytorch:
        ## Load data.
        data = [open("./ngram_authorship_train/"+filename, "r").read().lower() for filename in authorlist]
        data = [re.sub(r'[^\w\s]', '', text).split() for text in data]  ## Remove punctuation and split the text.
        print("Data size for each author:", [len(text) for text in data])

        ## Create the vocabulary. I am simple creating a vocab with everything in the data.
        vocabs = [sorted(set(text)) for text in data]
        print("Vocab size:", [len(vocab) for vocab in vocabs])

        word2idxs = [{word:i for i,word in enumerate(vocab)} for vocab in vocabs]
        idx2words = [{i:word for i,word in enumerate(vocab)} for vocab in vocabs]

        ## Make a encoder for each author. Encoder takes sentence and converts it to a list of indices.
        encoders = [lambda sentence, word2idx=word2idx: [word2idx[word] for word in sentence.split()] for word2idx in word2idxs]
        decoders = [lambda idxs, idx2word=idx2word: [idx2word[idx] for idx in idxs] for idx2word in idx2words]

        ## Convert the data into a torch tensor.
        data_tensors = [torch.tensor([encoder(sentence) for sentence in author], dtype=torch.long) for encoder, author in zip(encoders, data)]
        data_tensors = [item.squeeze(-1) for item in data_tensors]

        ## Split the data into training and testing data.
        training_tensors = [data[:int(len(data)*0.9)] for data in data_tensors]
        testing_tensors = [data[int(len(data)*0.9):] for data in data_tensors]

        ## Define batchloader
        # batch_size = 64
        # block_size = 8

        def get_batch(split, model_idx):
            data_for_batch = training_tensors[model_idx] if split == "train" else testing_tensors[model_idx]
            ix = torch.randint(len(data_for_batch) - block_size, (batch_size,))
            x = torch.stack([data_for_batch[i:i+block_size] for i in ix])
            y = torch.stack([data_for_batch[i+1:i+block_size+1] for i in ix])
            return x, y

        bigramModels =[BiGramModel(len(vocab)) for vocab in vocabs]

        optimizers = [torch.optim.Adam(model.parameters(), lr=1e-3) for model in bigramModels]

        ## Send the models to GPU.
        bigramModels = [model.to("cuda") for model in bigramModels]

        batch_size  = 256
        block_size = 32
        ## Train the models for 100 steps.
        for i, (model, optimizer) in enumerate(zip(bigramModels, optimizers)):
            for steps in range(10000):
                xb, yb = get_batch("train", i)
                xb, yb = xb.to("cuda"), yb.to("cuda")
                logits, loss = model(xb, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if steps % 100 == 0:
                    print(f"Step: {steps} Loss: {loss.item()}")

        ## Find index of I in the vocabs. To start the generation.
        idxs = [encode("i") for encode in encoders]

        idxs = [torch.tensor([idx], dtype=torch.long).to("cuda") for idx in idxs]
        texts = [model.generate(idx, 25)[0].squeeze(-1).tolist() for idx,model in zip(idxs,bigramModels)]
        for i, text in enumerate(texts):
            print(f"Generated text for {authorlist[i].split('.')[0]}: {decoders[i](text)}")