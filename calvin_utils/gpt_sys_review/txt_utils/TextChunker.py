
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
        self.chunk_metadata = []
        self.debug = debug
        self.chunk_tokens_dict = {}

        if self.debug:
            print(f"Initialized with token limit: {self.token_limit}")
    
    def chunk_text(self):
        """
        Splits the text into smaller segments based on the token limit.
        """
        import re
        
        # Use a more sophisticated approach to track dates line-by-line to avoid DOB
        lines = self.text.split('\n')
        
        if self.debug:
            print(f"Total lines to process: {len(lines)}")
        
        current_chunk = []
        current_chunk_tokens = 0
        current_date = "Unknown Date"

        for line in lines:
            # RPDR files often have the report date in the header row or near 'Encounter Date:'
            # We specifically look for these to avoid picking up DOB (Date of Birth)
            date_match = re.search(r'(?:Encounter Date:|Report_Date_Time,)\s*(\d{1,2}/\d{1,2}/\d{4})', line, re.IGNORECASE)
            if date_match:
                current_date = date_match.group(1)
            
            # Split line into words for token counting
            words = line.split()
            for word in words:
                tokens_in_word = 1 # Simple word count for token approximation + overhead
                
                if current_chunk_tokens + tokens_in_word <= self.token_limit:
                    current_chunk.append(word)
                    current_chunk_tokens += tokens_in_word
                else:
                    self.chunks.append(' '.join(current_chunk))
                    self.chunk_metadata.append({'date': current_date})
                    
                    context_str = f"[CONTINUED FROM REPORT DATE: {current_date}]"
                    current_chunk = [context_str, word]
                    current_chunk_tokens = len(context_str.split()) + 1 + tokens_in_word
            
            # Add a small token cost for the newline
            current_chunk.append('\n')
            current_chunk_tokens += 1

        if current_chunk:
            self.chunks.append(' '.join(current_chunk))
            self.chunk_metadata.append({'date': current_date})
            self.chunk_tokens_dict[len(self.chunks) - 1] = current_chunk_tokens
                
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

    def get_chunk_metadata(self):
        """
        Returns the list of metadata for each chunk.
        
        Returns:
        - list: List containing dictionaries of metadata for each chunk.
        """
        return self.chunk_metadata
    
    