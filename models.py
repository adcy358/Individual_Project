import torch 
from torch import nn 
import torch.nn.functional as F


class GRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(GRUEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers)

    def forward(self, input_seq, hidden):
        embedded = self.embedding(input_seq)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self, sequence_length):
        return torch.zeros(self.num_layers, sequence_length, self.hidden_size)
    

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers)

    def forward(self, input_seq, hidden):
        embedded = self.embedding(input_seq)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def init_hidden(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.hidden_size),
                torch.zeros(self.num_layers, sequence_length, self.hidden_size))
    
    
class GRUDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1):
        super(GRUDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden):
        embedded = self.embedding(input_seq)
        output, hidden = self.gru(embedded, hidden)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, sequence_length):
        return torch.zeros(self.num_layers, sequence_length, self.hidden_size)
    
    
class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden):
        embedded = self.embedding(input_seq)
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.hidden_size),
                torch.zeros(self.num_layers, sequence_length, self.hidden_size))
    
    
class Attention(nn.Module):
    # The attention mechanism will be applied between the decoder's hidden state and the encoder outputs.
    
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_weights = nn.Linear(hidden_size * 2, 1)

    def forward(self, decoder_hidden, encoder_outputs):
        
        # it repeats the decoder hidden state along the time dimension 
        # to match the sequence length of encoder outputs. 
        seq_len = encoder_outputs.size(0)
        repeated_decoder_hidden = decoder_hidden.expand(seq_len, -1, -1)
        
        # it obtains the energy score
        energy = torch.cat((repeated_decoder_hidden, encoder_outputs), dim=2)
        
        # it converts the energy tensor into a scalar value (i.e. attention score) for each time step.
        attention_scores = F.softmax(self.attention_weights(energy), dim=0)
        
        # it calculates the new context vector
        context_vector = torch.sum(attention_scores * encoder_outputs, dim=0)
        
        # Expand the context vector to match the batch size
        context_vector_expanded = context_vector.unsqueeze(0).expand(decoder_hidden.size(0), -1, -1)
        
        return context_vector_expanded, attention_scores
    
    
class GRUDecoderWithAttention(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1):
        super(GRUDecoderWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input_seq, hidden, encoder_outputs):
        embedded = self.embedding(input_seq)
        output, hidden = self.gru(embedded, hidden)

        # Calculate attention
        context_vector, attention_scores = self.attention(output, encoder_outputs)

        # Concatenate the context vector with the decoder output
        output = torch.cat((output, context_vector), dim=2)

        # Pass through the linear layer to get final output
        output = self.fc(output)

        return output, hidden, attention_scores

    def init_hidden(self, sequence_length):
        return torch.zeros(self.num_layers, sequence_length, self.hidden_size)