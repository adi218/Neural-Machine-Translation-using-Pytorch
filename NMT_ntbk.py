#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os

source_lang = 'en'
target_lang = 'vi'
data_dir = 'data/'


# In[2]:


def load_train(folder="./data/", rows=100000):
    for file in os.listdir(folder):
        file_path = os.path.join(os.path.abspath(folder), file)
        if file_path.__contains__("train"):
            if file_path.endswith(source_lang):
                file_en = open(file_path)
                dataset_en = _read_file(file_en)
            elif file_path.endswith(target_lang):
                file_vi = open(file_path)
                dataset_vi = _read_file(file_vi)
    if rows != -1:
        return [[dataset_en[i], dataset_vi[i]] for i in range(len(dataset_en))]
    return dataset_en, dataset_vi


# In[3]:


def _read_file(file):

    lines = file.readlines()
    lst_lines = [x.strip() for x in lines]
    return lst_lines


# In[4]:


train_data = load_train()


# In[5]:


corpus = pd.DataFrame(train_data) 


# In[6]:


# corpus[corpus[0].map(len) < 100]


# In[7]:


# corpus[corpus[1].map(len) < 100]


# In[8]:


SOS_token = '<start>'
EOS_token = '<end>'
UNK_token = '<unk>'
PAD_token = '<pad>'

SOS_idx = 0
EOS_idx = 1
UNK_idx = 2
PAD_idx = 3

class Vocab:
    def __init__(self):
        self.index2word = {
            SOS_idx: SOS_token,
            EOS_idx: EOS_token,
            UNK_idx: UNK_token,
            PAD_idx: PAD_token
        }
        self.word2index = {v: k for k, v in self.index2word.items()}

    def index_words(self, words):
        for word in words:
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            n_words = len(self)
            self.word2index[word] = n_words
            self.index2word[n_words] = word

    def __len__(self):
        assert len(self.index2word) == len(self.word2index)
        return len(self.index2word)

    def unidex_words(self, indices):
        return [self.index2word[i] for i in indices]

    def to_file(self, filename):
        values = [w for w, k in sorted(list(self.word2index.items())[5:])]
        with open(filename, 'w') as f:
            f.write('\n'.join(values))

    @classmethod
    def from_file(cls, filename):
        vocab = Vocab()
        with open(filename, 'r') as f:
            words = [l.strip() for l in f.readlines()]
            vocab.index_words(words)


# In[9]:


import nltk
import pandas as pd

max_length = 400
min_word_count = 1

tokenizers = {
    'en': nltk.tokenize.WordPunctTokenizer().tokenize,
    'vi': nltk.tokenize.WordPunctTokenizer().tokenize
}

def preprocess_corpus(sents, tokenizer, min_word_count):
    n_words = {}

    sents_tokenized = []
    for sent in sents:
        sent_tokenized = [w.lower() for w in tokenizer(sent)]

        sents_tokenized.append(sent_tokenized)

        for word in sent_tokenized:
            if word in n_words:
                n_words[word] += 1
            else:
                n_words[word] = 1

    for i, sent_tokenized in enumerate(sents_tokenized):
        sent_tokenized = [t if n_words[t] >= min_word_count else UNK_token for t in sent_tokenized]
        sents_tokenized[i] = sent_tokenized

    return sents_tokenized

def read_vocab(sents):
    vocab = Vocab()
    for sent in sents:
        vocab.index_words(sent)

    return vocab

source_sents = preprocess_corpus(corpus[0], tokenizers[source_lang], min_word_count)
target_sents = preprocess_corpus(corpus[1], tokenizers[target_lang], min_word_count)
print("preprocessing complete")

source_vocab = read_vocab(source_sents)
target_vocab = read_vocab(target_sents)
print("vocab created")
target_vocab.to_file(os.path.join(data_dir, '{}.vocab.txt'.format(target_lang)))
source_vocab.to_file(os.path.join(data_dir, '{}.vocab.txt'.format(source_lang)))
print("vocab saved to file")
print('Corpus length: {}\nSource vocabulary size: {}\nTarget vocabulary size: {}'.format(
    len(source_sents), len(source_vocab.word2index), len(target_vocab.word2index)
))
examples = list(zip(source_sents, target_sents))[80:90]
for source, target in examples:
    print('Source: "{}", target: "{}"'.format(' '.join(source), ' '.join(target)))


# In[10]:


import numpy as np

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

source_length = len(source_sents)
inidices = np.random.permutation(source_length)

training_indices = inidices[:int(source_length*0.94)]
dev_indices = inidices[int(source_length*0.8):int(source_length*0.99)]
test_indices = inidices[int(source_length*0.99):]

training_source = [source_sents[i] for i in training_indices]
dev_source = [source_sents[i] for i in dev_indices]
test_source = [source_sents[i] for i in test_indices]

training_target = [target_sents[i] for i in training_indices]
dev_target = [target_sents[i] for i in dev_indices]
test_target = [target_sents[i] for i in test_indices]

# Unwrap training examples
training_t = []
training_s = []
for source, tt in zip(training_source, training_target):
    training_t.append(tt)
    training_s.append(source)

training_source = training_s
training_target = training_t


# In[11]:


print(training_t[0])
print(training_s[0])


# In[12]:



import torch

def indexes_from_sentence(vocab, sentence):
    return [vocab.word2index[word] for word in sentence]

def tensor_from_sentence(vocab, sentence, max_seq_length):
#     print(sentence)
#     print("sentence over")
    indexes = indexes_from_sentence(vocab, sentence)
    indexes.append(EOS_idx)
    indexes.insert(0, SOS_idx)
    # we need to have all sequences the same length to process them in batches
    if len(indexes) < max_seq_length:
        indexes += [PAD_idx] * (max_seq_length - len(indexes))
    tensor = torch.LongTensor(indexes)
    return tensor

def tensors_from_pair(source_sent, target_sent, max_seq_length):
    source_tensor = tensor_from_sentence(source_vocab, source_sent, max_seq_length).unsqueeze(1)
    target_tensor = tensor_from_sentence(target_vocab, target_sent, max_seq_length).unsqueeze(1)
    return (source_tensor, target_tensor)

max_seq_length = max_length + 2  # 2 for EOS_token and SOS_token

training = []
for source_sent, target_sent in zip(training_source, training_target):
    training.append(tensors_from_pair(source_sent, target_sent, max_seq_length))



# In[13]:


x_training, y_training = zip(*training)


# In[14]:


x_training = list(x_training)
y_training = list(y_training)
new_x = []
new_y = []
for i in range(len(x_training)):
    if x_training[i].size()[0] == 402:
        new_x.append(x_training[i])
        new_y.append(y_training[i])
    else:
        print(i)
x_training = tuple(new_x)
y_training = tuple(new_y)


# In[15]:


new_x = []
new_y = []
for i in range(len(dev_source)):
    if len(dev_source[i]) <= 402:
        new_x.append(dev_source[i])
        new_y.append(dev_target[i])
    else:
        print(i)
dev_source = tuple(new_x)
dev_target = tuple(new_y)


# In[16]:


new_x = []
new_y = []
for i in range(len(y_training)):
    if y_training[i].size()[0] == 402:
        new_x.append(x_training[i])
        new_y.append(y_training[i])
    else:
        print(i)
x_training = tuple(new_x)
y_training = tuple(new_y)


# In[17]:


for i in range(len(x_training)):
    if x_training[i].size()[0] != 402:
        print(x_training[i].size()[0])


# In[18]:


x_training = torch.transpose(torch.cat(x_training, dim=-1), 1, 0)
y_training = torch.transpose(torch.cat(y_training, dim=-1), 1, 0)
torch.save(x_training, os.path.join(data_dir, 'x_training.bin'))
torch.save(y_training, os.path.join(data_dir, 'y_training.bin'))

x_development = []
for source_sent in dev_source:
    tensor = tensor_from_sentence(source_vocab, source_sent, max_seq_length).unsqueeze(1)
    x_development.append(tensor)

x_development = torch.transpose(torch.cat(x_development, dim=-1), 1, 0)
torch.save(x_development, os.path.join(data_dir, 'x_development.bin'))

x_test = []
for source_sent in test_source:
    tensor = tensor_from_sentence(source_vocab, source_sent, max_seq_length).unsqueeze(1)
    x_test.append(tensor)

x_test = torch.transpose(torch.cat(x_test, dim=-1), 1, 0)
torch.save(x_test, os.path.join(data_dir, 'x_test.bin'))

USE_CUDA = True
device="cpu"
if USE_CUDA:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_training = x_training.cuda()
    y_training = y_training.cuda()
    x_development = x_development.cuda()
    x_test = x_test.cuda()


# In[19]:


import torch.nn as nn
import torch.nn.init as init

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        init.normal_(self.embedding.weight, 0.0, 0.2)

        self.lstm = nn.LSTM(
            hidden_size,
            int(hidden_size/2),  # Bi-directional processing will ouput vectors of double size, therefore I reduced output dimensionality
            num_layers=n_layers,
            batch_first=True,  # First dimension of input tensor will be treated as a batch dimension
            bidirectional=True
        )

    # word_inputs: (batch_size, seq_length), h: (h_or_c, layer_n_direction, batch, seq_length)
    def forward(self, word_inputs, hidden):         
        # embedded (batch_size, seq_length, hidden_size)
        embedded = self.embedding(word_inputs)
        # output (batch_size, seq_length, hidden_size*directions)
        # hidden (h: (num_layers*directions, batch_size, hidden_size),
        #         c: (num_layers*directions, batch_size, hidden_size))
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def init_hidden(self, batches):
        hidden = torch.zeros(2, self.n_layers*2, batches, int(self.hidden_size/2), device=device)
        if USE_CUDA: hidden = hidden.cuda()
        return hidden


# In[20]:


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers=1):
        super(DecoderRNN, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        init.normal_(self.embedding.weight, 0.0, 0.2)

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=n_layers, batch_first=True, bidirectional=False)

    def forward(self, word_inputs, hidden):
        # Note: we run this one by one
        # embedded (batch_size, 1, hidden_size)
        embedded = self.embedding(word_inputs).unsqueeze_(1)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden


# In[21]:


class Seq2seq(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, hidden_size, n_layers):
        super(Seq2seq, self).__init__()

        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.encoder = EncoderRNN(input_vocab_size, hidden_size, self.n_layers).to(device)
        self.decoder = DecoderRNN(output_vocab_size, hidden_size, self.n_layers).to(device)

        self.W = nn.Linear(hidden_size, output_vocab_size).to(device)
        init.normal_(self.W.weight, 0.0, 0.2)

        self.softmax = nn.Softmax()

    def _forward_encoder(self, x):
        batch_size = x.shape[0]
        init_hidden = self.encoder.init_hidden(batch_size)
        encoder_outputs, encoder_hidden = self.encoder(x, init_hidden)
        encoder_hidden_h, encoder_hidden_c = encoder_hidden

        self.decoder_hidden_h = encoder_hidden_h.permute(1,0,2).reshape(batch_size, self.n_layers, self.hidden_size).permute(1,0,2)
        self.decoder_hidden_c = encoder_hidden_c.permute(1,0,2).reshape(batch_size, self.n_layers, self.hidden_size).permute(1,0,2)
        return self.decoder_hidden_h, self.decoder_hidden_c

    def forward_train(self, x, y):
        decoder_hidden_h, decoder_hidden_c = self._forward_encoder(x)

        H = []
        for i in range(y.shape[1]):
            input = y[:, i]
            decoder_output, decoder_hidden = self.decoder(input, (decoder_hidden_h, decoder_hidden_c))
            decoder_hidden_h, decoder_hidden_c = decoder_hidden
            # h: (batch_size, vocab_size)
            h = self.W(decoder_output.squeeze(1))
            # h: (batch_size, vocab_size, 1)
            H.append(h.unsqueeze(2))

        # H: (batch_size, vocab_size, seq_len)
        return torch.cat(H, dim=2)

    def forward(self, x):
        decoder_hidden_h, decoder_hidden_c = self._forward_encoder(x)

        current_y = SOS_idx
        result = [current_y]
        counter = 0
        while current_y != EOS_idx and counter < 100:
            input = torch.tensor([current_y], device = device)
            decoder_output, decoder_hidden = self.decoder(input, (decoder_hidden_h, decoder_hidden_c))
            decoder_hidden_h, decoder_hidden_c = decoder_hidden
            # h: (vocab_size)
            h = self.W(decoder_output.squeeze(1)).squeeze(0)
            y = self.softmax(h)
            _, current_y = torch.max(y, dim=0)
            current_y = current_y.item()
            result.append(current_y)
            counter += 1

        return result


# In[22]:


from torch.optim import Adam

model = Seq2seq(len(source_vocab), len(target_vocab), 300, 1)
optim = Adam(model.parameters(), lr=0.0001)


# In[23]:


import math

def batch_generator(batch_indices, batch_size):
    batches = math.ceil(len(batch_indices)/batch_size)
    for i in range(batches):
        batch_start = i*batch_size
        batch_end = (i+1)*batch_size
        if batch_end > len(batch_indices):
            yield batch_indices[batch_start:]
        else:
            yield batch_indices[batch_start:batch_end]


# In[24]:


cross_entropy = nn.CrossEntropyLoss()


# In[25]:


from nltk.translate.bleu_score import corpus_bleu

def bleu(n):
    weights = [1.0/n]*n + [0.0]*(4-n)
    return lambda list_of_references, list_of_hypothesis: corpus_bleu(list_of_references, list_of_hypothesis, weights)

def accuracy(list_of_references, list_of_hypothesis):
    total = 0.0
    for references, hypothesis in zip(list_of_references, list_of_hypothesis):
        total += 1.0 if tuple(hypothesis) in set(references) else 0.0
    return total / len(list_of_references)

score_functions = {'BLEU-{}'.format(i):bleu(i) for i in range(1, 5)}
score_functions['Accuracy'] = accuracy

def score(model, X, target, desc='Scoring...'):
    scores = {name:0.0 for name in score_functions.keys()}
    length = len(target)
    list_of_hypothesis = []
    for i, x in tqdm(enumerate(X),
                     desc=desc,
                     total=length):
        y = model(x.unsqueeze(0))
        hypothesis = target_vocab.unidex_words(y[1:-1])  # Remove SOS and EOS from y
        list_of_hypothesis.append(hypothesis)

    for name, func in score_functions.items():
        score = func(target, list_of_hypothesis)
        scores[name] = score

    return scores


# In[26]:


from tqdm import tqdm_notebook as tqdm

BATCH_SIZE = 4
total_batches = int(len(x_training)/BATCH_SIZE) + 1
indices = list(range(len(x_training)))

early_stop_after = 10
early_stop_counter = 0
best_model = None

best_score = 0.0
scoring_metric = 'BLEU-1'
scores_history = []
loss_history = []

for epoch in range(100):
    # Training
    total_loss = 0.0
    for step, batch in tqdm(enumerate(batch_generator(indices, BATCH_SIZE)),
                            desc='Training epoch {}'.format(epoch+1),
                            total=total_batches):
        x = x_training[batch, :]
        # y for teacher forcing is all sequence without a last element
        y_tf = y_training[batch, :-1]
        # y for loss calculation is all sequence without a last element
        y_true = y_training[batch, 1:]
        # (batch_size, vocab_size, seq_length)
        H = model.forward_train(x, y_tf)
        loss = cross_entropy(H, y_true)
        print(step, " ", loss)
        assert loss.item() > 0

        optim.zero_grad()
        loss.backward()
        optim.step()

        total_loss += loss.item()

    loss_history.append(total_loss/total_batches)
    print('Epoch {} training is finished, loss: {:.4f}'.format(epoch+1, total_loss/total_batches))

    desc = 'Validating epoch {}'.format(epoch+1)
    scores = score(model, x_development, dev_target, desc=desc)
    scores_str = '\n'.join(['{}: {:.4f}'.format(name, score) for name, score in scores.items()])
    scores_history.append(scores)

    print ('Epoch {} validation is finished.\n{}'.format(
        epoch+1, scores_str
    ))

    metric = scores[scoring_metric]

    # Early Stop
    if metric > best_score:
        early_stop_counter = 0
        print('The best model is found, resetting early stop counter.')
        best_score = metric
        best_model = model
    else:
        early_stop_counter += 1
        print('No improvements for {} epochs.'.format(early_stop_counter))
        if early_stop_counter >= early_stop_after:
            print('Early stop!')
            break
torch.save(best_model.state_dict(), "./model/model")

# In[ ]:




