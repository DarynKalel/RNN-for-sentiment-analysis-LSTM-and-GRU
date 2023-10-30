# RNN-for-sentiment-analysis-LSTM-and-GRU

In this project, I will explore how to develop a simple Recurrent Neural Network (RNN) for sentiment analysis. I will use the IMDB dataset---it contains the text of some reviews and the sentiment given by their authors (either positive or negative). The input to the RNN is the sequence of words that compose a review, so the learning task consists in predicting the overall sentiment of the review.

# Dataset

In this project I used an IMDB dataset. The dataset contains 50,000 movie reviews from IMDB, labeled by sentiment (positive/negative). As usual, for speed and efficiency, I will use only a subset of the dataset. Reviews have been preprocessed, and each review is encoded as a sequence of word indexes. I load the data from the PyTorch database and then split the data into train, validation and test set.

# Model Definition

The first layer is an Embedding layer, with input_dim=vocab_dim and output_dim=10. The model will gradually learn to represent each of the 10,000 words as a 10-dimensional vector. So the next layer will receive 3D batches of shape (batch size, 500, 10)
The second layer is the recurrent one. In particular, in this case, we use a [RNN]
The output layer

# Model Comparison
In the next cell, I define simple RNN used for binary classification. The class has two main methods, the constructor (init()) and the forward() method.

In the constructor, the input parameters are used to define the layers and hyperparameters of the RNN. The layers that are defined include an embedding layer (self.embedding), a recurrent layer (self.rnn), and a linear layer (self.linear). The constructor also sets up various parameters such as the embedding input size, the embedding output size, the hidden size, the number of layers, the batch size, the RNN type, and whether or not the RNN is bidirectional.

The forward() method takes a batch of input data (x) and applies the layers defined in the constructor in a specific sequence. First, the input is passed through the embedding layer to create embeddings of the input tokens. These embeddings are then permuted to be of the correct shape for the RNN layer, which expects inputs of the form (seq_len, batch_size, H_in). The RNN layer is then applied to these embeddings, producing both the RNN output (rnn_out) and the last hidden state (self.last_hidden). Finally, the output of the RNN is passed through a linear layer and flattened to produce the final output of the network, which is a sigmoid activation function applied to a tensor of shape (batch_size). This output is then returned.

# LSTM and GRU
I implemented LSTM model and GRU model, similar to the previous one that, instead of exploiting the RNN layer, used an LSTM and GRU layers. Trained them and plotted the values of accuracy and loss. Compared with the performance of bi-directional LSTM. In the end experimented with transfromer for the same task.

# The structure of the transformer is defined as follows:

A multi-head attention layer
Dropout operation (dropout_att)
Layer Normalization (layernorm_att)
A feedforward Neural Network, Sequential, and Dense layer
Dropout operation (dropout_fnn)
Layer Normalization (layernorm_fnn) that has in input the summation of the attention layer output and the feedforward NN output.

See the results onn the script.
