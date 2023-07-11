import pandas as pd
import numpy as np
import re
import torch
from collections import defaultdict, Counter
from pprint import pprint
import warnings

class Vocab:
    
    """The Vocab class is responsible for:
    Creating dataset's vocabulary.
    Filtering dataset in terms of the rare words occurrence and sentences lengths.
    Mapping words to their numerical representation (word2index) and reverse (index2word).
    Enabling the use of pre-trained word vectors.
    
    SOURCE: https://github.com/radoslawkrolikowski/sentiment-analysis-pytorch/blob/master/5_biGRU_with_Glove_vectors.ipynb

    Parameters
    ----------
    dataset : pandas.DataFrame or numpy.ndarray
        Pandas or numpy dataset containing in the first column input strings to process and target non-string 
        variable as last column.
    target_col: int, optional (default=None)
        Column index refering to targets strings to process.
    word2index: dict, optional (default=None)
        Specify the word2index mapping.
    sos_token: str, optional (default='<SOS>')
        Start of sentence token.
    eos_token: str, optional (default='<EOS>')
        End of sentence token.
    unk_token: str, optional (default='<UNK>')
        Token that represents unknown words.
    pad_token: str, optional (default='<PAD>')
        Token that represents padding.
    min_word_count: float, optional (default=5)
        Specify the minimum word count threshold to include a word in vocabulary if value > 1 was passed.
        If min_word_count <= 1 then keep all words whose count is greater than the quantile=min_word_count
        of the count distribution.
    max_vocab_size: int, optional (default=None)
        Maximum size of the vocabulary.
    max_seq_len: float, optional (default=0.8)
        Specify the maximum length of the sequence in the dataset, if max_seq_len > 1. If max_seq_len <= 1 then set
        the maximum length to value corresponding to quantile=max_seq_len of lengths distribution. Trimm all
        sequences whose lengths are greater than max_seq_len.
    use_pretrained_vectors: boolean, optional (default=False)
        Whether to use pre-trained Glove vectors.
    glove_path: str, optional (default='Glove/')
        Path to the directory that contains files with the Glove word vectors.
    glove_name: str, optional (default='glove.6B.100d.txt')
        Name of the Glove word vectors file. Available pretrained vectors:
        glove.6B.50d.txt
        glove.6B.100d.txt
        glove.6B.200d.txt
        glove.6B.300d.txt
        glove.twitter.27B.50d.txt
        To use different word vectors, load their file to the vectors directory (Glove/).
    weights_file_name: str, optional (default='Glove/weights.npy')
        The path and the name of the numpy file to which save weights vectors.

    Raises
    -------
    ValueError('Use min_word_count or max_vocab_size, not both!')
        If both: min_word_count and max_vocab_size are provided.
    FileNotFoundError
        If the glove file doesn't exists in the given directory.

    """
    
    
    def __init__(self, dataset, target_col=None, word2index=None, sos_token='<SOS>', eos_token='<EOS>', unk_token='<UNK>',
             pad_token='<PAD>', min_word_count=5, max_vocab_size=None, max_seq_len=0.8,
             use_pretrained_vectors=False, glove_path='Glove/', glove_name='glove.6B.100d.txt',
             weights_file_name='Glove/weights.npy'):
        
        # Convert pandas dataframe to numpy.ndarray
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.to_numpy()
        
        self.dataset = dataset
        self.target_col = target_col
        
        if self.target_col:
            self.y_lengths = []
            
        self.x_lengths = []
        self.word2idx_mapping = word2index
        
        # Define word2idx and idx2word as empty dictionaries
        if self.word2idx_mapping:
            self.word2index = self.word2idx_mapping
        else:
            self.word2index = defaultdict(dict)
            self.index2word = defaultdict(dict)            
        
        # Instantiate special tokens
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        
        # Instantiate min_word_count, max_vocab_size and max_seq_len
        self.min_word_count = min_word_count
        self.max_vocab_size = max_vocab_size
        self.max_seq_len = max_seq_len
        
        self.use_pretrained_vectors = use_pretrained_vectors
        
        if self.use_pretrained_vectors: 
            self.glove_path = glove_path
            self.glove_name = glove_name
            self.weights_file_name = weights_file_name
        
        self.build_vocab()
        
        
    def build_vocab(self):
        """Build the vocabulary, filter dataset sequences and create the weights matrix if specified.
        
        """
        # Create a dictionary that maps words to their count
        self.word_count = self.word2count()

        # Trim the vocabulary
        # Get rid of out-of-vocabulary words from the dataset
        if self.min_word_count or self.max_vocab_size:
            self.trimVocab()
            self.trimDatasetVocab()

        # Trim sequences in terms of length
        if self.max_seq_len:
            if self.x_lengths:
                self.trimSeqLen()

            else:
                # Calculate sequences lengths
                self.x_lengths = [len(seq.split()) for seq in self.dataset[:, 0]]
                
                if self.target_col:
                    self.y_lengths = [len(seq.split()) for seq in self.dataset[:, self.target_col]]
                    
                self.trimSeqLen()                

                
        # Map each tokens to index
        if not self.word2idx_mapping:
            self.mapWord2index()
               
        # Crate index2word mapping
        self.index2word = {index: word for word, index in self.word2index.items()}
        
        # Map dataset tokens to indices
        self.mapWords2indices()
        
        # Create weights matrix based on Glove vectors
        if self.use_pretrained_vectors:
            self.glove_vectors()       
        
            
    def word2count(self):
        """Count the number of words occurrences.
        
        """
        # Instantiate the Counter object
        word_count = Counter()

        # Iterate through the dataset and count tokens
        for line in self.dataset[:, 0]:
            word_count.update(line.split())
            
            # Include strings from target column
            if self.target_col:
                for line in self.dataset[:, self.target_col]:
                    word_count.update(line.split())
            
        return word_count
    

    def trimVocab(self):
        """Trim the vocabulary in terms of the minimum word count or the vocabulary maximum size.
        
        """
        # Trim the vocabulary in terms of the minimum word count
        if self.min_word_count and not self.max_vocab_size:
            # If min_word_count <= 1, use the quantile approach
            if self.min_word_count <= 1:
                # Create the list of words count
                word_stat = [count for count in self.word_count.values()]
                # Calculate the quantile of words count
                quantile = int(np.quantile(word_stat, self.min_word_count))
                print('Trimmed vocabulary using as mininum count threashold: quantile({:3.2f}) = {}'.\
                      format(self.min_word_count, quantile))
                # Filter words using quantile threshold
                self.trimmed_word_count = {word: count for word, count in self.word_count.items() if count >= quantile}
            # If min_word_count > 1 use standard approach
            else:
                # Filter words using count threshold
                self.trimmed_word_count = {word: count for word, count in self.word_count.items()\
                                   if count >= self.min_word_count}
                print('Trimmed vocabulary using as minimum count threashold: count = {:3.2f}'.format(self.min_word_count))
                     
        # Trim the vocabulary in terms of its maximum size
        elif self.max_vocab_size and not self.min_word_count:
            self.trimmed_word_count = {word: count for word, count in self.word_count.most_common(self.max_vocab_size)}
            print('Trimmed vocabulary using maximum size of: {}'.format(self.max_vocab_size))
        else:
            raise ValueError('Use min_word_count or max_vocab_size, not both!')
            
        print('{}/{} tokens has been retained'.format(len(self.trimmed_word_count.keys()),
                                                     len(self.word_count.keys())))

    
    def trimDatasetVocab(self):
        """Get rid of rare words from the dataset sequences.
        
        """
        for row in range(self.dataset.shape[0]):
            trimmed_x = [word for word in self.dataset[row, 0].split() if word in self.trimmed_word_count.keys()]
            self.x_lengths.append(len(trimmed_x))
            self.dataset[row, 0] = ' '.join(trimmed_x)
        print('Trimmed input strings vocabulary')
                            
        if self.target_col:
            for row in range(self.dataset.shape[0]):
                trimmed_y = [word for word in self.dataset[row, self.target_col].split()\
                             if word in self.trimmed_word_count.keys()]
                self.y_lengths.append(len(trimmed_y))
                self.dataset[row, self.target_col] = ' '.join(trimmed_y)
            print('Trimmed target strings vocabulary')
            
                
    def trimSeqLen(self):
        """Trim dataset sequences in terms of the length.
        
        """
        if self.max_seq_len <= 1:
            x_threshold = int(np.quantile(self.x_lengths, self.max_seq_len)) 
            if self.target_col:
                y_threshold = int(np.quantile(self.y_lengths, self.max_seq_len)) 
        else:
            x_threshold = self.max_seq_len
            if self.target_col:
                y_threshold =  self.max_seq_len
        
        if self.target_col:      
            for row in range(self.dataset.shape[0]):
                x_truncated = ' '.join(self.dataset[row, 0].split()[:x_threshold])\
                if self.x_lengths[row] > x_threshold else self.dataset[row, 0]
                
                # Add 1 if the EOS token is going to be added to the sequence
                self.x_lengths[row] = len(x_truncated.split()) if not self.eos_token else \
                                      len(x_truncated.split()) + 1
                
                self.dataset[row, 0] = x_truncated
                
                y_truncated = ' '.join(self.dataset[row, self.target_col].split()[:y_threshold])\
                if self.y_lengths[row] > y_threshold else self.dataset[row, self.target_col]
                
                # Add 1 or 2 to the length to inculde special tokens
                y_length = len(y_truncated.split())
                if self.sos_token and not self.eos_token:
                    y_length = len(y_truncated.split()) + 1
                elif self.eos_token and not self.sos_token:
                    y_length = len(y_truncated.split()) + 1
                elif self.sos_token and self.eos_token:
                    y_length = len(y_truncated.split()) + 2
                    
                self.y_lengths[row] = y_length
                
                self.dataset[row, self.target_col] = y_truncated
                
            print('Trimmed input sequences lengths to the length of: {}'.format(x_threshold))
            print('Trimmed target sequences lengths to the length of: {}'.format(y_threshold))
            
        else:
            for row in range(self.dataset.shape[0]):

                x_truncated = ' '.join(self.dataset[row, 0].split()[:x_threshold])\
                if self.x_lengths[row] > x_threshold else self.dataset[row, 0]
                
                # Add 1 if the EOS token is going to be added to the sequence
                self.x_lengths[row] = len(x_truncated.split()) if not self.eos_token else \
                                      len(x_truncated.split()) + 1
                
                self.dataset[row, 0] = x_truncated
                
            print('Trimmed input sequences lengths to the length of: {}'.format(x_threshold))
                
        
    def mapWord2index(self):
        """Populate vocabulary word2index dictionary.
        
        """
        # Add special tokens as first elements in word2index dictionary
        token_count = 0
        for token in [self.pad_token, self.sos_token, self.eos_token, self.unk_token]:
            if token:
                self.word2index[token] = token_count
                token_count += 1
        
        # If vocabulary is trimmed, use trimmed_word_count
        if self.min_word_count or self.max_vocab_size:
            for key in self.trimmed_word_count.keys():
                self.word2index[key] = token_count
                token_count += 1
            
        # If vocabulary is not trimmed, iterate through dataset    
        else:
            for line in self.dataset.iloc[:, 0]:
                for word in line.split():
                    if word not in self.word2index.keys():
                        self.word2index[word] = token_count
                        token_count += 1
            # Include strings from target column
            if self.target_col:
                for line in self.dataset.iloc[:, self.target_col]:
                    for word in line.split():
                        if word not in self.word2index.keys():
                            self.word2index[word] = token_count
                            token_count += 1
                            
        self.word2index.default_factory = lambda: self.word2index[self.unk_token]
                            
        
    def mapWords2indices(self):
        """Iterate through the dataset to map each word to its corresponding index.
        Use special tokens if specified.
        
        """
        for row in range(self.dataset.shape[0]):
            words2indices = []
            for word in self.dataset[row, 0].split():
                words2indices.append(self.word2index[word])
                    
            # Append the end of the sentence token
            if self.eos_token:
                words2indices.append(self.word2index[self.eos_token])
                
            self.dataset[row, 0] = np.array(words2indices)
                
        # Map strings from target column
        if self.target_col:
            for row in range(self.dataset.shape[0]):
                words2indices = []
                
                # Insert the start of the sentence token
                if self.sos_token:
                    words2indices.append(self.word2index[self.sos_token])
                    
                for word in self.dataset[row, self.target_col].split():
                    words2indices.append(self.word2index[word])

                        
                # Append the end of the sentence token
                if self.eos_token:
                    words2indices.append(self.word2index[self.eos_token])
                    
                self.dataset[row, self.target_col] = np.array(words2indices)
           
        print('Mapped words to indices')

    
    def glove_vectors(self):
        """ Read glove vectors from a file, create the matrix of weights mapping vocabulary tokens to vectors.
        Save the weights matrix to the numpy file.
        
        """
        # Load Glove word vectors to the pandas dataframe
        try:
            gloves = pd.read_csv(self.glove_path + self.glove_name, sep=" ", quoting=3, header=None, index_col=0)
        except FileNotFoundError:
            print('File: {} not found in: {} directory'.format(self.glove_name, self.glove_path))
            
        # Map Glove words to vectors
        print('Start creating glove_word2vector dictionary')
        self.glove_word2vector = gloves.T.to_dict(orient='list')
        
        # Extract embedding dimension
        emb_dim = int(re.findall('\d+' ,self.glove_name)[-1])
        # Length of the vocabulary
        matrix_len = len(self.word2index)
        # Initialize the weights matrix
        weights_matrix = np.zeros((matrix_len, emb_dim))
        words_found = 0

        # Populate the weights matrix
        for word, index in self.word2index.items():
            try: 
                weights_matrix[index] = np.array(self.glove_word2vector[word])
                words_found += 1
            except KeyError:
                # If vector wasn't found in Glove, initialize random vector
                weights_matrix[index] = np.random.normal(scale=0.6, size=(emb_dim, ))
         
        # Save the weights matrix into numpy file
        np.save(self.weights_file_name, weights_matrix, allow_pickle=False)
        
        # Delete glove_word2vector variable to free the memory
        del self.glove_word2vector
                        
        print('Extracted {}/{} of pre-trained word vectors.'.format(words_found, matrix_len))
        print('{} vectors initialized to random numbers'.format(matrix_len - words_found))
        print('Weights vectors saved into {}'.format(self.weights_file_name))
                
class BatchIterator:
    """The BatchIterator class is responsible for:
    Sorting dataset examples.
    Generating batches.
    Sequence padding.
    Enabling BatchIterator instance to iterate through all batches.

    Parameters
    ----------
    dataset : pandas.DataFrame or numpy.ndarray
        If vocab_created is False, pass Pandas or numpy dataset containing in the first column input strings
        to process and target non-string variable as last column. Otherwise pass vocab.dataset object.
    batch_size: int, optional (default=None)
        The size of the batch. By default use batch_size equal to the dataset length.
    vocab_created: boolean, optional (default=True)
        Whether the vocab object is already created.
    vocab: Vocab object, optional (default=None)
        Use if vocab_created = True, pass the vocab object.
    target_col: int, optional (default=None)
        Column index refering to targets strings to process.
    word2index: dict, optional (default=None)
        Specify the word2index mapping.
    sos_token: str, optional (default='<SOS>')
        Use if vocab_created = False. Start of sentence token.
    eos_token: str, optional (default='<EOS>')
        Use if vocab_created = False. End of sentence token.
    unk_token: str, optional (default='<UNK>')
        Use if vocab_created = False. Token that represents unknown words.
    pad_token: str, optional (default='<PAD>')
        Use if vocab_created = False. Token that represents padding.
    min_word_count: float, optional (default=5)
        Use if vocab_created = False. Specify the minimum word count threshold to include a word in vocabulary
        if value > 1 was passed. If min_word_count <= 1 then keep all words whose count is greater than the
        quantile=min_word_count of the count distribution.
    max_vocab_size: int, optional (default=None)
        Use if vocab_created = False. Maximum size of the vocabulary.
    max_seq_len: float, optional (default=0.8)
        Use if vocab_created = False. Specify the maximum length of the sequence in the dataset, if 
        max_seq_len > 1. If max_seq_len <= 1 then set the maximum length to value corresponding to
        quantile=max_seq_len of lengths distribution. Trimm all sequences whose lengths are greater
        than max_seq_len.
    use_pretrained_vectors: boolean, optional (default=False)
        Use if vocab_created = False. Whether to use pre-trained Glove vectors.
    glove_path: str, optional (default='Glove/')
        Use if vocab_created = False. Path to the directory that contains files with the Glove word vectors.
    glove_name: str, optional (default='glove.6B.100d.txt')
        Use if vocab_created = False. Name of the Glove word vectors file. Available pretrained vectors:
        glove.6B.50d.txt
        glove.6B.100d.txt
        glove.6B.200d.txt
        glove.6B.300d.txt
        glove.twitter.27B.50d.txt
        To use different word vectors, load their file to the vectors directory (Glove/).
    weights_file_name: str, optional (default='Glove/weights.npy')
        Use if vocab_created = False. The path and the name of the numpy file to which save weights vectors.

    Raises
    -------
    ValueError('Use min_word_count or max_vocab_size, not both!')
        If both: min_word_count and max_vocab_size are provided.
    FileNotFoundError
        If the glove file doesn't exist in the given directory.
    TypeError('Cannot convert to Tensor. Data type not recognized')
        If the data type of the sequence cannot be converted to the Tensor.

    Yields
    ------
    dict
        Dictionary that contains variables batches.

    """
        
        
    def __init__(self, dataset, batch_size=None, vocab_created=False, vocab=None, target_col=None, word2index=None,
             sos_token='<SOS>', eos_token='<EOS>', unk_token='<UNK>', pad_token='<PAD>', min_word_count=5,
             max_vocab_size=None, max_seq_len=0.8, use_pretrained_vectors=False, glove_path='Glove/',
             glove_name='glove.6B.100d.txt', weights_file_name='Glove/weights.npy'):    
    
        # Create vocabulary object
        if not vocab_created:
            self.vocab = Vocab(dataset, target_col=target_col, word2index=word2index, sos_token=sos_token, eos_token=eos_token,
                               unk_token=unk_token, pad_token=pad_token, min_word_count=min_word_count,
                               max_vocab_size=max_vocab_size, max_seq_len=max_seq_len,
                               use_pretrained_vectors=use_pretrained_vectors, glove_path=glove_path,
                               glove_name=glove_name, weights_file_name=weights_file_name)
            
            # Use created vocab.dataset object
            self.dataset = self.vocab.dataset      
        
        else:
            # If vocab was created then dataset should be the vocab.dataset object
            self.dataset = dataset
            self.vocab = vocab
            
        self.target_col = target_col 
        
        self.word2index = self.vocab.word2index
            
        # Define the batch_size
        if batch_size:
            self.batch_size = batch_size
        else:
            # Use the length of dataset as batch_size
            self.batch_size = len(self.dataset)
                
        self.x_lengths = np.array(self.vocab.x_lengths)
        
        if self.target_col:
            self.y_lengths = np.array(self.vocab.y_lengths)
            
        self.pad_token = self.vocab.word2index[pad_token]
            
        self.sort_and_batch()

        
    def sort_and_batch(self):
        """ Sort examples within entire dataset, then perform batching and shuffle all batches.

        """
        # Extract row indices sorted according to lengths
        if not self.target_col:
            sorted_indices = np.argsort(self.x_lengths)
        else:
            sorted_indices = np.lexsort((self.y_lengths, self.x_lengths))
        
        # Sort all sets
        self.sorted_dataset = self.dataset[sorted_indices[::-1]]
        self.sorted_x_lengths = np.flip(self.x_lengths[sorted_indices])
        
        if self.target_col:
            self.sorted_target = self.sorted_dataset[:, self.target_col]
            self.sorted_y_lengths = np.flip(self.x_lengths[sorted_indices])
        else:
            self.sorted_target = self.sorted_dataset[:, -1]
        
        # Initialize input, target and lengths batches
        self.input_batches = [[] for _ in range(self.sorted_dataset.shape[1]-1)]
        
        self.target_batches, self.x_len_batches = [], []

        self.y_len_batches = [] if self.target_col else None
        
        # Create batches
        for i in range(self.sorted_dataset.shape[1]-1):
            # The first column contains always sequences that should be padded.
            if i == 0:
                self.create_batches(self.sorted_dataset[:, i], self.input_batches[i], pad_token=self.pad_token)
            else:
                self.create_batches(self.sorted_dataset[:, i], self.input_batches[i])
                
        if self.target_col:
            self.create_batches(self.sorted_target, self.target_batches, pad_token=self.pad_token)
            self.create_batches(self.sorted_y_lengths, self.y_len_batches)
        else:
            self.create_batches(self.sorted_target, self.target_batches)
        
        self.create_batches(self.sorted_x_lengths, self.x_len_batches)
        
        # Shuffle batches
        self.indices = np.arange(len(self.input_batches[0]))
        np.random.shuffle(self.indices)
        
        for j in range(self.sorted_dataset.shape[1]-1):
            self.input_batches[j] = [self.input_batches[j][i] for i in self.indices]
        
        self.target_batches = [self.target_batches[i] for i in self.indices]
        self.x_len_batches = [self.x_len_batches[i] for i in self.indices]
        
        if self.target_col:
            self.y_len_batches = [self.y_len_batches[i] for i in self.indices]
        
        print('Batches created')
        
        
    def create_batches(self, sorted_dataset, batches, pad_token=-1):
        """ Convert each sequence to pytorch Tensor, create batches and pad them if required.
        
        """
        # Calculate the number of batches
        n_batches = int(len(sorted_dataset)/self.batch_size)
#         print(n_batches)
#         print(len(sorted_dataset))
#         print(self.batch_size)
        # Create list of batches
#         print(sorted_dataset[i*self.batch_size:(i+1)*self.batch_size].copy() for i in range(n_batches+1))
        
        list_of_batches = np.array([sorted_dataset[i*self.batch_size:(i+1)*self.batch_size].copy()\
                                    for i in range(n_batches+1)], dtype = object)

        # Convert each sequence to pytorch Tensor
        for batch in list_of_batches:
            tensor_batch = []
            tensor_type = None
            for seq in batch:
                # Check seq data type and convert to Tensor
                if isinstance(seq, np.ndarray):
                    tensor = torch.LongTensor(seq)
                    tensor_type = 'int'
                elif isinstance(seq, np.integer):
                    tensor = torch.LongTensor([seq])
                    tensor_type = 'int'
                elif isinstance(seq, np.float64):
                    tensor = torch.FloatTensor([seq])
                    tensor_type = 'float'
                elif isinstance(seq, int):
                    tensor = torch.LongTensor([seq])
                    tensor_type = 'int'
                elif isinstance(seq, float):
                    tensor = torch.FloatTensor([seq])
                    tensor_type = 'float'
                else:
                    raise TypeError('Cannot convert to Tensor. Data type not recognized')

                tensor_batch.append(tensor)
            if pad_token != -1:
                # Pad required sequences
                pad_batch = torch.nn.utils.rnn.pad_sequence(tensor_batch, batch_first=True)
                batches.append(pad_batch)
            else:
                if tensor_type == 'int':
                    batches.append(torch.LongTensor(tensor_batch))
                else:
                    batches.append(torch.FloatTensor(tensor_batch))

                
    def __iter__(self):
        """ Iterate through batches.
        
        """
        # Create a dictionary that holds variables batches to yield
        to_yield = {}
        
        # Iterate through batches
        for i in range(len(self.input_batches[0])):
            feat_list = []
            for j in range(1, len(self.input_batches)):
                feat = self.input_batches[j][i].type(torch.FloatTensor).unsqueeze(1)
                feat_list.append(feat)
                
            if feat_list:
                input_feat = torch.cat(feat_list, dim=1)
                to_yield['input_feat'] = input_feat

            to_yield['input_seq'] = self.input_batches[0][i]

            to_yield['target'] = self.target_batches[i]
            to_yield['x_lengths'] = self.x_len_batches[i]
            
            if self.target_col:
                to_yield['y_length'] = self.y_len_batches[i]


            yield to_yield
            
            
    def __len__(self):
        """ Return iterator length.
        
        """
        return len(self.input_batches[0])
        