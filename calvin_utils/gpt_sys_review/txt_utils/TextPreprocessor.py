import os
import re
from tqdm import tqdm

class TextPreprocessor:
    """
    A class to preprocess text files in a directory.

    Attributes:
    - input_dir (str): The directory containing the text files to be preprocessed.
    - output_dir (str): The directory where the preprocessed text files will be saved.

    Methods:
    - preprocess_text: Applies various preprocessing steps to a given text.
    - remove_non_ascii: Removes non-ASCII characters from the text.
    - process_files: Reads each text file from the input directory, applies preprocessing, and saves it to the output directory.
    """

    def __init__(self, input_dir):
        """
        Initializes the TextPreprocessor class with input and output directories.

        Parameters:
        - input_dir (str): Path to the directory containing the text files to be preprocessed.
        - output_dir (str): Path to the directory where the preprocessed text files will be saved.
        """
        self.input_dir = input_dir
        self.output_dir = input_dir + '_preprocessed'
        self._prep_out_dir()
        
    def _prep_out_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def preprocess_text(text):
        """
        Applies various preprocessing steps to a given text.

        Parameters:
        - text (str): The text to be preprocessed.

        Returns:
        - str: The preprocessed text.
        """
        text = re.sub(r'([,.:;])', r'\1 ', text)
        text = re.sub(r'([(])', r' \1', text)
        text = re.sub(r'([)])', r'\1 ', text)
        text = re.sub(r'(?<![a-zA-Z0-9-])-(?![a-zA-Z0-9-])', r' - ', text)
        return text

    @staticmethod
    def remove_non_ascii(text):
        """
        Removes non-ASCII characters from the text.

        Parameters:
        - text (str): The text from which non-ASCII characters will be removed.

        Returns:
        - str: The text with non-ASCII characters removed.
        """
        return re.sub(r'[^\x00-\x7F]+', ' ', text)

    def _extract_text(self, input_filepath, encoding='utf-8'):
        with open(input_filepath, 'r', encoding='utf-8') as input_file:
            return input_file.read()

    def process_files(self):
        """
        Reads each text file from the input directory, applies preprocessing, and saves it to the output directory.
        """
        raw_txt_files = os.listdir(self.input_dir)                          # get target files
        raw_txt_files = [f for f in raw_txt_files if f.endswith('.txt')]    # make sure only to process .txt files
        for filename in tqdm(raw_txt_files, desc='Preprocessing text files'):
            input_filepath = os.path.join(self.input_dir, filename)
            output_filepath = os.path.join(self.output_dir, filename)

            try: 
                original_text = self._extract_text(input_filepath)
            except Exception as e:
                print(f'Skipping {filename} due to error: {e}')
                continue
            
            # Apply preprocessing
            preprocessed_text = self.preprocess_text(original_text)
            cleaned_text = self.remove_non_ascii(preprocessed_text)
            
            # Save the preprocessed text
            with open(output_filepath, 'w', encoding='utf-8') as output_file:
                output_file.write(cleaned_text)
        return self.output_dir

class TextChunker:
    """
    A class to chunk a given text into smaller segments based on a token limit.
    """
    
    def __init__(self, text, token_limit, debug=False):
        """
        Initializes the TextChunker class with the text and token limit.
        
        Parameters:
        - text (str): The text to be chunked.
        - token_limit (int): The maximum number of tokens allowed in each chunk.
        - debug (bool): Flag to enable debug print statements.
        """
        self.text = text
        self.token_limit = token_limit
        self.chunks = []
        self.debug = debug
        self.chunk_tokens_dict = {}

        if self.debug:
            print(f"Initialized with token limit: {self.token_limit}")
    
    def chunk_text(self):
        """
        Splits the text into smaller segments based on the token limit.
        """
        words = self.text.split()
        
        if self.debug:
            print(f"Total words to process: {len(words)}")
            if all(word == '' for word in words):
                print("Warning: All words are empty spaces.")
        
        current_chunk = []
        current_chunk_tokens = 0

        for word in words:
            tokens_in_word = len(word.split()) + 1
            if self.debug:
                print(f"Found {tokens_in_word} tokens in word: '{word}'")
            if current_chunk_tokens + tokens_in_word <= self.token_limit:
                current_chunk.append(word)
                current_chunk_tokens += tokens_in_word
                if self.debug:
                    print(f"Added word to current chunk, total tokens now: {current_chunk_tokens}")
            else:
                self.chunks.append(' '.join(current_chunk))
                if self.debug:
                    print(f"Chunk completed, appended to chunks list.")
                current_chunk = [word]
                current_chunk_tokens = tokens_in_word
                if self.debug:
                    print(f"Started new chunk with word: '{word}', total tokens now: {current_chunk_tokens}")

        if current_chunk:
            self.chunks.append(' '.join(current_chunk))
            self.chunk_tokens_dict[len(self.chunks) - 1] = current_chunk_tokens
            if self.debug:
                print(f"Final chunk added to chunks list.")
                
    def chunk_tokens(self):
        for key, value in self.chunk_tokens_dict.items():
            print(f'chunk {key} tokens: {value}')
            
    def get_chunks(self):
        """
        Returns the list of generated text chunks.
        
        Returns:
        - list: List containing the generated text chunks.
        """
        return self.chunks
