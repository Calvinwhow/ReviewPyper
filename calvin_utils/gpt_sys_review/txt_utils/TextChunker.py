
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
        Splits the text into smaller segments based on the token limit, 
        after sorting all individual reports chronologically.
        """
        import re
        from datetime import datetime
        
        # 1. Identify individual reports and their dates
        # RPDR files use | delimiters (now ,) and headers like "Report_Date_Time"
        # We split by the pattern of the start of a report header
        lines = self.text.split('\n')
        reports = []
        current_report_lines = []
        current_report_date = datetime(1900, 1, 1) # Default very old date
        
        # Trigger pattern for a new report header in RPDR (columns often separated by , after our earlier replacement)
        # We look for the common MRN_Type or EMPI column start or the Encounter Date line
        for line in lines:
            date_match = re.search(r'(?:Encounter Date:|Report_Date_Time,)\s*(\d{1,2}/\d{1,2}/\d{4})', line, re.IGNORECASE)
            
            # If we find a new date-defining line, we treat it as a potential start of a new report if it's not DOB
            if date_match and "DOB" not in line:
                if current_report_lines:
                    reports.append((current_report_date, "\n".join(current_report_lines)))
                
                try: current_report_date = datetime.strptime(date_match.group(1), "%m/%d/%Y")
                except: current_report_date = datetime(1900, 1, 1)
                
                current_report_lines = [line]
            else:
                current_report_lines.append(line)
        
        if current_report_lines:
            reports.append((current_report_date, "\n".join(current_report_lines)))
        
        # 2. Sort reports chronologically
        reports.sort(key=lambda x: x[0])
        
        # 3. Chunk the sorted reports
        if self.debug:
            print(f"Sorted {len(reports)} reports for process.")
            
        self.chunks = []
        self.chunk_metadata = []
        current_chunk = []
        current_chunk_tokens = 0
        
        for date, report_text in reports:
            date_str = date.strftime("%m/%d/%Y") if date != datetime(1900, 1, 1) else "Unknown Date"
            
            # Split reports into words for token counting
            words = report_text.split()
            for word in words:
                tokens_in_word = 1
                
                if current_chunk_tokens + tokens_in_word <= self.token_limit:
                    current_chunk.append(word)
                    current_chunk_tokens += tokens_in_word
                else:
                    self.chunks.append(' '.join(current_chunk))
                    self.chunk_metadata.append({'date': date_str})
                    
                    context_str = f"[CONTINUED FROM REPORT DATE: {date_str}]"
                    current_chunk = [context_str, word]
                    current_chunk_tokens = len(context_str.split()) + 1 + tokens_in_word
            
            # Ensure the metadata at the end of a report belongs to its date
            if current_chunk:
                # Add a marker for end of a report to prevent bleeding across reports without a date update
                current_chunk.append('\n')
                current_chunk_tokens += 1

        if current_chunk:
            # The final chunk date should reflect the date of the last report processed in it
            self.chunks.append(' '.join(current_chunk))
            last_date = reports[-1][0].strftime("%m/%d/%Y") if reports else "Unknown Date"
            self.chunk_metadata.append({'date': last_date})
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
    
    