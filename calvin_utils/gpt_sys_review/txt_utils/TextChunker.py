
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
        self.text = text.replace('|', ',')  # Replace "|", since we use it as a separator for the answers
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
        import re
        words = self.text.split()
        
        if self.debug:
            print(f"Total words to process: {len(words)}")
            if all(word == '' for word in words):
                print("Warning: All words are empty spaces.")
        
        current_chunk = []
        current_chunk_tokens = 0
        current_date = "Unknown Date"

        for word in words:
            # Track the date context from the pipe-separated (now comma-separated) RPDR header
            date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', word)
            if date_match and ',' in word:
                current_date = date_match.group(1)

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
                
                context_str = f"[CONTINUED FROM REPORT DATE: {current_date}]"
                current_chunk = [context_str, word]
                current_chunk_tokens = len(context_str.split()) + 1 + tokens_in_word
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
    
    