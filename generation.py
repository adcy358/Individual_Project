import numpy as np
import torch 
import torch.nn.functional as F


# compute perplexity
def perplexity_val(probs, n):
    #epsilon = 1e-10  # Small constant to prevent zero probabilities
    #smoothed_probs = probs + epsilon
    return -1/float(n) * np.log(probs)


def construct_poem(word_array): 
    lines = []
    line = []

    for word in word_array:
        if word == "<SOV>":
            line = []
        elif word == "<EOV>":
            lines.append(' '.join(line))
        else:
            line.append(word)

    poem = '\n'.join(lines) 
    
    return poem


# Generation loop
def generate_poem_GRU(encoder, decoder, dataset, start_token, end_token, 
                   max_length=50, temperature=1.0, top_k=10, encoder_type='GRU', 
                   encoder_ckpt = "saved_models/encoder_model.pt", 
                   decoder_ckpt = "saved_models/encoder_model.pt"
):
    
    # Load the saved models
    encoder.load_state_dict(torch.load(encoder_ckpt))
    decoder.load_state_dict(torch.load(decoder_ckpt))
    
    encoder.eval()
    decoder.eval()
    
    #compute perplexity
    perplexity = 1
    words_in_poem = 0
    poem_perplexity = []
    
    # init variables
    lines = 0
    poem_array = []
    
    with torch.no_grad():
        input_seq = torch.tensor([start_token]).unsqueeze(0)
        
        if encoder_type == 'GRU':
            hidden = encoder.init_hidden(1)
            # Forward pass - Encoder
            encoder_outputs, hidden = encoder(input_seq, hidden)
        
        elif encoder_type == 'LSTM': 
            hidden, cell_state = encoder.init_hidden(1)
            # Forward pass - Encoder
            encoder_outputs, (hidden, cell_state) = encoder(input_seq, (hidden, cell_state))

        # Initialize the hidden state of the decoder with the final encoder hidden state
        decoder_hidden = hidden

        # Initialize the generated poem
        poem_idx = [start_token]

        # Generate poem word by word
        for _ in range(max_length):
            decoder_input = torch.tensor([poem_idx[-1]]).unsqueeze(0)

            # Forward pass - Decoder
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            
            # Apply temperature to the logits
            logits = decoder_output / temperature

            # Top-k sampling
            top_k_probs, top_k_words = logits.topk(top_k)
            top_k_probs = top_k_probs.squeeze()
            top_k_words = top_k_words.squeeze()

            # Randomly select from the top-k words
            word_idx = torch.multinomial(torch.exp(top_k_probs), 1).item()
            chosen_word = top_k_words[word_idx].item()
            
            poem_idx.append(chosen_word)
                
            # Get the probabilities of the words using softmax
            word_probs = torch.nn.functional.softmax(decoder_output, dim=2).squeeze(0)
            p = word_probs[0, word_idx].item()
            words_in_poem += 1
            perplexity *= p
    
            #if word_idx == end_token: 
                #lines+=1
                
            if words_in_poem == max_length:
                poem_idx.append(1)
                for word_index in poem_idx:
                    word = dataset.index_to_word[word_index]
                    poem_array.append(word)
                
                #print(poem_array)
                generated_poem = construct_poem(poem_array)
                
                # compute the total perplexity of the generated text
                poem_perplexity = perplexity_val(perplexity,words_in_poem)
                
                break

    return generated_poem, poem_perplexity



def generate_poem_LSTM(encoder, decoder, dataset, start_token, end_token, 
                   max_length=50, temperature=1.0, top_k=10, 
                   encoder_ckpt = "saved_models/encoder_3.pt", 
                   decoder_ckpt = "saved_models/decoder_3.pt"
):
    
    # Load the saved models
    encoder.load_state_dict(torch.load(encoder_ckpt))
    decoder.load_state_dict(torch.load(decoder_ckpt))
    
    encoder.eval()
    decoder.eval()
    
    #compute perplexity
    perplexity = 1
    words_in_poem = 0
    poem_perplexity = []
    
    # init variables
    lines = 0
    poem_array = []
    
    with torch.no_grad():
        input_seq = torch.tensor([start_token]).unsqueeze(0)
        hidden, cell_state = encoder.init_hidden(1)
        
        # Forward pass - Encoder
        encoder_outputs, (hidden, cell_state) = encoder(input_seq, (hidden, cell_state))

        # Initialize the hidden state and cell state of the decoder with the final encoder hidden and cell state
        decoder_hidden = hidden
        decoder_cell_state = cell_state

        # Initialize the generated poem
        poem_idx = [start_token]

        # Generate poem word by word
        for _ in range(max_length):
            decoder_input = torch.tensor([poem_idx[-1]]).unsqueeze(0)

            # Forward pass - Decoder
            decoder_output, (decoder_hidden, decoder_cell_state) = decoder(decoder_input, (decoder_hidden, decoder_cell_state))
            
            # Apply temperature to the logits
            logits = decoder_output / temperature

            # Top-k sampling
            top_k_probs, top_k_words = logits.topk(top_k)
            top_k_probs = top_k_probs.squeeze()
            top_k_words = top_k_words.squeeze()

            # Randomly select from the top-k words
            word_idx = torch.multinomial(torch.exp(top_k_probs), 1).item()
            chosen_word = top_k_words[word_idx].item()
            
            poem_idx.append(chosen_word)
                
            # Get the probabilities of the words using softmax
            word_probs = torch.nn.functional.softmax(decoder_output, dim=2).squeeze(0)
            p = word_probs[0, word_idx].item()
            words_in_poem += 1
            perplexity *= p
                
            if words_in_poem == max_length:
                poem_idx.append(1)
                for word_index in poem_idx:
                    word = dataset.index_to_word[word_index]
                    poem_array.append(word)
                
                #print(poem_array)
                generated_poem = construct_poem(poem_array)
                
                # compute the total perplexity of the generated text
                poem_perplexity = perplexity_val(perplexity,words_in_poem)
                
                break

    return generated_poem, poem_perplexity



def generate_poem_GRU_with_attention(encoder, decoder, dataset, start_token, end_token, 
                   max_length=50, temperature=1.0, top_k=10, 
                   encoder_ckpt = "saved_models/encoder_model.pt", 
                   decoder_ckpt = "saved_models/encoder_model.pt"
):
    
    # Load the saved models
    encoder.load_state_dict(torch.load(encoder_ckpt))
    decoder.load_state_dict(torch.load(decoder_ckpt))
    
    encoder.eval()
    decoder.eval()
    
    #compute perplexity
    perplexity = 1
    words_in_poem = 0
    poem_perplexity = []
    
    # init variables
    lines = 0
    poem_array = []
    
    with torch.no_grad():
        input_seq = torch.tensor([start_token]).unsqueeze(0)

        hidden, cell_state = encoder.init_hidden(1)
        # Forward pass - Encoder
        encoder_outputs, (hidden, cell_state) = encoder(input_seq, (hidden, cell_state))
        
        # Initialize the hidden state of the decoder with the final encoder hidden state
        decoder_hidden = hidden
        
        # Initialize the generated poem
        poem_idx = [start_token]

        # Generate poem word by word
        for _ in range(max_length):
            decoder_input = torch.tensor([poem_idx[-1]]).unsqueeze(0)

            # Forward pass - Decoder with Attention
            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
            
            # Apply temperature to the logits
            logits = decoder_output / temperature

            # Top-k sampling
            top_k_probs, top_k_words = logits.topk(top_k)
            top_k_probs = top_k_probs.squeeze()
            top_k_words = top_k_words.squeeze()

            # Randomly select from the top-k words
            word_idx = torch.multinomial(torch.exp(top_k_probs), 1).item()
            chosen_word = top_k_words[word_idx].item()
            
            poem_idx.append(chosen_word)
                
            # Get the probabilities of the words using softmax
            word_probs = torch.nn.functional.softmax(decoder_output, dim=2).squeeze(0)
            p = word_probs[0, word_idx].item()
            words_in_poem += 1
            perplexity *= p
    
            #if word_idx == end_token: 
                #lines+=1
                
            if words_in_poem == max_length:
                poem_idx.append(1)
                for word_index in poem_idx:
                    word = dataset.index_to_word[word_index]
                    poem_array.append(word)
                
                #print(poem_array)
                generated_poem = construct_poem(poem_array)
                
                # compute the total perplexity of the generated text
                poem_perplexity = perplexity_val(perplexity,words_in_poem)
                
                break

    return generated_poem, poem_perplexity