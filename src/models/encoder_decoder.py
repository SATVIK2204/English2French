import torch
from torch import nn
import torch.nn as nn
from English2French.src.models.encoder import Encoder
from English2French.src.models.decoder import Decoder
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()


def loss_function(real, pred):
    """Calculate how wrong the model is."""
    # Use mask to only consider non-zero inputs in the loss
    mask = real.ge(1).float().to(device)

    loss_ = criterion(pred, real) * mask
    return torch.mean(loss_)


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        inputs_vocab_size,
        targets_vocab_size,
        hidden_size,
        embedding_dim,
        batch_size,
        targets_start_idx,
        targets_stop_idx,
    ):
        super(EncoderDecoder, self).__init__()
        self.batch_size = batch_size
        self.targets_start_idx = targets_start_idx
        self.targets_stop_idx = targets_stop_idx

        self.encoder = Encoder(
            inputs_vocab_size, embedding_dim, hidden_size, batch_size
        ).to(device)

        self.decoder = Decoder(
            targets_vocab_size, embedding_dim, hidden_size, hidden_size, batch_size
        ).to(device)

    def predict(self, inputs, lengths):
        self.batch_size = inputs.size(0)

        encoder_output, encoder_hidden = self.encoder(
            inputs.to(device),
            lengths,
        )
        decoder_hidden = encoder_hidden

        # Initialize the input of the decoder to be <SOS>
        decoder_input = torch.LongTensor(
            [[self.targets_start_idx]] * self.batch_size,
        )

        # Output predictions instead of loss
        output = []
        for _ in range(20):
            predictions, decoder_hidden, _ = self.decoder(
                decoder_input.to(device),
                decoder_hidden.to(device),
                encoder_output.to(device),
            )
            prediction = torch.multinomial(F.softmax(predictions, dim=1), 1)
            decoder_input = prediction

            prediction = prediction.item()
            output.append(prediction)

            if prediction == self.targets_stop_idx:
                return output

        return output

    def forward(self, inputs, targets, lengths):
        self.batch_size = inputs.size(0)

        encoder_output, encoder_hidden = self.encoder(
            inputs.to(device),
            lengths,
        )
        decoder_hidden = encoder_hidden

        # Initialize the input of the decoder to be <SOS>
        decoder_input = torch.LongTensor(
            [[self.targets_start_idx]] * self.batch_size,
        )

        # Use teacher forcing to train the model. Instead of feeding the model's
        # own predictions to itself, feed the target token at every timestep.
        # This leads to faster convergence
        loss = 0
        for timestep in range(1, targets.size(1)):
            predictions, decoder_hidden, _ = self.decoder(
                decoder_input.to(device),
                decoder_hidden.to(device),
                encoder_output.to(device),
            )
            decoder_input = targets[:, timestep].unsqueeze(1)

            loss += loss_function(targets[:, timestep], predictions)

        return loss / targets.size(1)
