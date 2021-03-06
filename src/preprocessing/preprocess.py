from nltk import wordpunct_tokenize
from torch.utils.data import Dataset
from collections import Counter

def tokenize(text):
    """Turn text into discrete tokens.

    Remove tokens that are not words.
    """
    text = text.lower()
    tokens = wordpunct_tokenize(text)

    # Only keep words
    tokens = [token for token in tokens
              if all(char.isalpha() for char in token)]

    return tokens


class PrepareTheData(Dataset):
    def __init__(self, df, max_vocab):
        self.max_vocab = max_vocab
        
        # Extra tokens to add
        self.padding_token = '<PAD>'
        self.start_of_sequence_token = '<SOS>'
        self.end_of_sequence_token = '<EOS>'
        self.unknown_word_token = '<UNK>'

        # Helper function
        self.flatten = lambda x: [sublst for lst in x for sublst in lst]
        
        # Tokenize inputs (English) and targets (French)
        self.tokenize_df(df)
        print('Tokenization Done')
        # To reduce computational complexity, replace rare words with <UNK>
        self.replace_rare_tokens(df)
        print('Replacing Done')
        # Prepare variables with mappings of tokens to indices
        self.create_token2idx(df)
        print('token2idx created')
        # Remove sequences with mostly <UNK>
        # df = self.remove_mostly_unk(df)
        
        # Every sequence (input and target) should start with <SOS>
        # and end with <EOS>
        self.add_start_and_end_to_tokens(df)
        print('Added Start and End tokens')
        
        # Convert tokens to indices
        self.tokens_to_indices(df)
        print('tokens_to_indices created')
        print('Presprocessing done')
        
    def __getitem__(self, idx):
        """Return example at index idx."""
        return self.indices_pairs[idx][0], self.indices_pairs[idx][1]
    
    def tokenize_df(self, df):
        """Turn inputs and targets into tokens."""
        df['english'] = df.english.apply(tokenize)
        df['french'] = df.french.apply(tokenize)
        
    def replace_rare_tokens(self, df):
        """Replace rare tokens with <UNK>."""
        common_tokens_inputs = self.get_most_common_tokens(
            df.english.tolist(),
        )
        common_tokens_targets = self.get_most_common_tokens(
            df.french.tolist(),
        )
        
        df.loc[:, 'english'] = df.english.apply(
            lambda tokens: [token if token in common_tokens_inputs 
                            else self.unknown_word_token for token in tokens]
        )
        df.loc[:, 'french'] = df.french.apply(
            lambda tokens: [token if token in common_tokens_targets
                            else self.unknown_word_token for token in tokens]
        )

    def get_most_common_tokens(self, tokens_series):
        """Return the max_vocab most common tokens."""
        all_tokens = self.flatten(tokens_series)
        # Substract 4 for <PAD>, <SOS>, <EOS>, and <UNK>
        common_tokens = set(list(zip(*Counter(all_tokens).most_common(
            self.max_vocab - 4)))[0])
        return common_tokens

    def remove_mostly_unk(self, df, threshold=0.99):
        """Remove sequences with mostly <UNK>."""
        calculate_ratio = (
            lambda tokens: sum(1 for token in tokens if token != '<UNK>')
            / len(tokens) > threshold
        )
        df = df[df.english.apply(calculate_ratio)]
        df = df[df.french.apply(calculate_ratio)]
        return df
        
    def create_token2idx(self, df):
        """Create variables with mappings from tokens to indices."""
        unique_tokens_inputs = set(self.flatten(df.english))
        unique_tokens_targets = set(self.flatten(df.french))
        
        for token in reversed([
            self.padding_token,
            self.start_of_sequence_token,
            self.end_of_sequence_token,
            self.unknown_word_token,
        ]):
            if token in unique_tokens_inputs:
                unique_tokens_inputs.remove(token)
            if token in unique_tokens_targets:
                unique_tokens_targets.remove(token)
                
        unique_tokens_inputs = sorted(list(unique_tokens_inputs))
        unique_tokens_targets = sorted(list(unique_tokens_targets))

        # Add <PAD>, <SOS>, <EOS>, and <UNK> tokens
        for token in reversed([
            self.padding_token,
            self.start_of_sequence_token,
            self.end_of_sequence_token,
            self.unknown_word_token,
        ]):
            
            unique_tokens_inputs = [token] + unique_tokens_inputs
            unique_tokens_targets = [token] + unique_tokens_targets
            
        self.token2idx_inputs = {token: idx for idx, token
                                 in enumerate(unique_tokens_inputs)}
        self.idx2token_inputs = {idx: token for token, idx
                                 in self.token2idx_inputs.items()}
        
        self.token2idx_targets = {token: idx for idx, token
                                  in enumerate(unique_tokens_targets)}
        self.idx2token_targets = {idx: token for token, idx
                                  in self.token2idx_targets.items()}
        
    def add_start_and_end_to_tokens(self, df):
        """Add <SOS> and <EOS> tokens to the end of every input and output."""
        df['english']=df.english.apply(lambda x: [self.start_of_sequence_token]+x+[self.end_of_sequence_token])
        df['french']=df.french.apply(lambda x: [self.start_of_sequence_token]+x+[self.end_of_sequence_token])
        
    def tokens_to_indices(self, df):
        """Convert tokens to indices."""
        df['english'] = df.english.apply(
            lambda tokens: [self.token2idx_inputs[token] for token in tokens])
        df['french'] = df.french.apply(
            lambda tokens: [self.token2idx_targets[token] for token in tokens])
             
        self.indices_pairs = list(zip(df.english, df.french))
        
    def __len__(self):
        return len(self.indices_pairs)