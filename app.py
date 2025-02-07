import streamlit as st
import torch
import pickle
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import os
from nltk.corpus import words
from transformers import BertTokenizer, BertModel, BertForSequenceClassification

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

class TransformerLSTMModel(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, lstm_layers, transformer_layers=1, dropout=0.5):
        super(TransformerLSTMModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.lstm = nn.LSTM(embedding_matrix.size(1), hidden_dim, lstm_layers, batch_first=True)
        transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=transformer_layers)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        transformer_out = self.transformer_encoder(lstm_out)
        pooled = torch.mean(transformer_out, dim=1)
        return self.sigmoid(self.fc(pooled))

class BERTSentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BERTSentimentClassifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=n_classes)

    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask)

class TransformerBiLSTMModel(nn.Module):
    def __init__(self, bert_model, hidden_dim, lstm_layers=4, transformer_layers=2, dropout=0.2):
        super(TransformerBiLSTMModel, self).__init__()
        self.bert = bert_model
        self.hidden_dim = hidden_dim
        self.bilstm = nn.LSTM(
            input_size=bert_model.config.hidden_size,  
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_dim * 2, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=transformer_layers)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, lengths):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            bert_embeddings = bert_output.last_hidden_state
        lengths_sorted, perm_idx = lengths.sort(0, descending=True)
        bert_embeddings_sorted = bert_embeddings[perm_idx]
        packed_input = pack_padded_sequence(bert_embeddings_sorted, lengths_sorted.cpu(), batch_first=True)
        lstm_out, _ = self.bilstm(packed_input)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        _, unsort_idx = perm_idx.sort(0)
        lstm_out = lstm_out[unsort_idx]
        transformer_out = self.transformer_encoder(lstm_out)
        pooled = torch.mean(transformer_out, dim=1)
        output = self.dropout(pooled)
        output = torch.relu(self.fc1(output))
        output = self.fc2(output)
        return self.sigmoid(output)
    
def preprocessed(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text, flags=re.MULTILINE)
    text = re.sub(r'#\w+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text, flags=re.MULTILINE).strip()
    return text



def tokenize_and_remove_stopwords(text):
    custom_stop_words = {'im', 'u', 'rt'}
    sentiment_critical = {
        'no', 'not', 'none', 'never', 'nothing', 'nowhere', 'neither', 'nor',
        'very', 'really', 'extremely', 'quite',
        'would', 'should', 'could', 'might', 'must',
        'but', 'however', 'although', 'though', 'despite', 'yet',
        'just', 'too', 'only', 'even', 'still'
    }
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english')).union(custom_stop_words) - sentiment_critical
    tokens = [token for token in tokens if (token not in stop_words or token in sentiment_critical)]
    return tokens

def lemmitize_words(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

def preprocess_text(text):
    text = preprocessed(text)
    text = tokenize_and_remove_stopwords(text)
    text = lemmitize_words(text)
    return text

def load_glove_embeddings(path, expected_dim):
    english_words = set(words.words())
    embedding_index = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in english_words:
                vector = torch.tensor([float(val) for val in values[1:]], dtype=torch.float32)
                if vector.size(0) == expected_dim:
                    embedding_index[word] = vector
                else:
                    print(f"Skipped word '{word}' due to unexpected dimension size: {vector.size(0)}")
    return embedding_index

# Create Embedding Matrix
def create_embedding_matrix(word_index, embeddings_index, dimension):
    embedding_matrix = torch.zeros(len(word_index) + 1, dimension)
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

@st.cache_resource
def load_resources():
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    dataset_path = 'dataset_tweets_preprocessed2.csv'

    # Load dataset
    if not os.path.exists(dataset_path):
        st.error(f"Dataset not found at {dataset_path}. Ensure the file is present.")
        st.stop()

    df = pd.read_csv(dataset_path, on_bad_lines='skip')
    
    df['processed_text_str'] = df['processed_text_str'].astype(str)
    df['target'] = df['target'].replace({4: 1})

    # Vectorizers
    if os.path.exists('BoW_vectorizer.pkl'):
        with open('BoW_vectorizer.pkl', 'rb') as f:
            BoW_vectorizer = pickle.load(f)
    else:
        BoW_vectorizer = CountVectorizer(max_features=30000, min_df=5, max_df=0.90)
        BoW_vectorizer.fit(df['processed_text_str'])
        with open('BoW_vectorizer.pkl', 'wb') as f:
            pickle.dump(BoW_vectorizer, f)

    if os.path.exists('Tfidf_vectorizer.pkl'):
        with open('Tfidf_vectorizer.pkl', 'rb') as f:
            Tfidf_vectorizer = pickle.load(f)
    else:
        Tfidf_vectorizer = TfidfVectorizer(max_features=30000, min_df=5, max_df=0.90)
        Tfidf_vectorizer.fit(df['processed_text_str'])
        with open('Tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(Tfidf_vectorizer, f)

    model_bow = LogisticRegressionModel(len(BoW_vectorizer.vocabulary_))
    model_bow.load_state_dict(torch.load('trained_model_bow.pth', map_location=torch.device('cpu')))
    model_bow.eval()

    model_tfidf = LogisticRegressionModel(len(Tfidf_vectorizer.vocabulary_))
    model_tfidf.load_state_dict(torch.load('trained_model_tfidf.pth', map_location=torch.device('cpu')))
    model_tfidf.eval()

    model_bilstm_transformer = TransformerBiLSTMModel(bert_model, hidden_dim=128, lstm_layers=4, transformer_layers=2, dropout=0.2)
    model_bilstm_transformer.load_state_dict(torch.load('Transformer_BiLSTM_BERT_model.pth', map_location=torch.device('cpu')))
    model_bilstm_transformer.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model_bert = BERTSentimentClassifier(n_classes=2)
    model_bert.load_state_dict(torch.load('BERT_model.pth', map_location=torch.device('cpu')))
    model_bert.eval()

    return BoW_vectorizer, Tfidf_vectorizer, model_bow, model_tfidf, model_bilstm_transformer,tokenizer, model_bert

BoW_vectorizer, Tfidf_vectorizer, model_bow, model_tfidf, model_bilstm_transformer,tokenizer, model_bert = load_resources()

st.title("🌟 Sentiment Analysis Application")
st.subheader("Enter Your Text Below 👇")
txt_input = st.text_area("", "Type here...")

st.subheader("Select Your Model 🎛")
model_choice = st.radio("", [
    "Logistic Regression (BoW)",
    "Logistic Regression (TF-IDF)",
    "Pretrained-BERT Model (Bert Embeddings)",
    "Transformer + BiLSTM (BERT Embeddings)",
], index=0)

if st.button("✨ Classify Now"):
    if txt_input.strip() == "":
        st.warning("⚠️ Please enter some text to classify.")
    else:
        preprocessed_text = preprocessed(txt_input)

        # Default values
        label = "Unknown"
        confidence = 0.0

        # Process for each model
        if model_choice == "Logistic Regression (BoW)":
            vectorizer = BoW_vectorizer
            model = model_bow
            input_vector = vectorizer.transform([preprocessed_text]).toarray()
            input_tensor = torch.FloatTensor(input_vector)
            with torch.no_grad():
                output = model(input_tensor)
                prediction = (output > 0.5).item()
                confidence = output.item()
                label = "Positive" if prediction == 1 else "Negative"

        elif model_choice == "Logistic Regression (TF-IDF)":
            vectorizer = Tfidf_vectorizer
            model = model_tfidf
            input_vector = vectorizer.transform([preprocessed_text]).toarray()
            input_tensor = torch.FloatTensor(input_vector)
            with torch.no_grad():
                output = model(input_tensor)
                prediction = (output > 0.5).item()
                confidence = output.item()
                label = "Positive" if prediction == 1 else "Negative"

        elif model_choice == "Transformer + BiLSTM (BERT Embeddings)":
            encoding = tokenizer.encode_plus(
                preprocessed_text,
                max_length=512,
                add_special_tokens=True,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]
            lengths = (attention_mask != 0).sum(dim=1)
            with torch.no_grad():
                output = model_bilstm_transformer(input_ids, attention_mask, lengths)
                prediction = (output > 0.5).item()
                confidence = output.item()
                label = "Positive" if prediction == 1 else "Negative"

        elif model_choice == "Pretrained-BERT Model (Bert Embeddings)":
            encoding = tokenizer.encode_plus(
                preprocessed_text,
                add_special_tokens=True,
                max_length=128,
                return_attention_mask=True,
                truncation=True,
                padding='max_length',
                return_tensors="pt",
            )
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            with torch.no_grad():
                outputs = model_bert(input_ids=input_ids, attention_mask=attention_mask)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, prediction].item()
                label = "Positive" if prediction == 1 else "Negative"

        else:
            st.error("⚠️ Unknown model choice. Please select a valid option.")
        
        if label != "Unknown":
            st.markdown(f"## 📝 Prediction: **{label}**")
            st.markdown(f"### 🔥 Confidence: **{confidence:.2f}**")
