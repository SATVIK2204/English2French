import torch
from torch import nn
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        decoder_hidden_size,
        encoder_hidden_size,
        batch_size,
    ):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = nn.GRU(
            self.embedding_dim + self.encoder_hidden_size,
            self.decoder_hidden_size,
            batch_first=True,
        )
        self.fc = nn.Linear(self.encoder_hidden_size, self.vocab_size)

        # Attention weights
        self.W1 = nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size)
        self.W2 = nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size)
        self.V = nn.Linear(self.encoder_hidden_size, 1)

    def forward(self, targets, hidden, encoder_output):
        self.batch_size = targets.size(0)

        # Switch the dimensions of sequence_length and batch_size
        encoder_output = encoder_output.permute(1, 0, 2)

        # Add an extra axis for a time dimension
        hidden_with_time_axis = hidden.permute(1, 0, 2)

        # Attention score (Bahdanaus)
        score = torch.tanh(self.W1(encoder_output) + self.W2(hidden_with_time_axis))

        # Attention weights
        attention_weights = torch.softmax(self.V(score), dim=1)

        # Find the context vectors
        context_vector = attention_weights * encoder_output
        context_vector = torch.sum(context_vector, dim=1)

        # Turn target indices into distributed embeddings
        x = self.embedding(targets)

        # Add the context representation to the target embeddings
        x = torch.cat((context_vector.unsqueeze(1), x), -1)

        # Apply the RNN
        output, state = self.gru(x, self.init_hidden())

        # Reshape the hidden states (output)
        output = output.view(-1, output.size(2))

        # Apply a linear layer
        x = self.fc(output)

        return x, state, attention_weights

    def init_hidden(self):
        # Randomly initialize the weights of the RNN
        return torch.randn(1, self.batch_size, self.decoder_hidden_size).to(device)
