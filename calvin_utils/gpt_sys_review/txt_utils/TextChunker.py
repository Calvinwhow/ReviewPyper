
class TextChunker:
    """
    A class to chunk a given text into smaller segments based on a token limit.
    """
    
    def __init__(self, text, token_limit, chunk_by_date=False,debug=False):
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
        self.chunk_by_date=chunk_by_date
        self.debug = debug
        self.chunk_tokens_dict = {}

        if self.debug:
            print(f"Initialized with token limit: {self.token_limit}")
    
    @staticmethod
    def count_tokens_in_word(text):
        """Counts the # of tokens in a word"""
        #TODO: This is a placeholder. currently we're just counting 1 word=1 token, 
        # but once we implement a real tokenizer we can use this to ensure consistency
        return 1
    
    def chunk_text(self):
        if self.chunk_by_date:
            self.chunk_text_by_date()
        else:
            self.chunk_text_by_tokenlimit()

    @staticmethod
    def reformat_date(date_str):
        """Reformats a date string from M/D/YYYY to YYYY-mm-dd for easier sorting.
           This format is also good for the final output, since it makes comparing dates
           easy for the eye."""
        if date_str.lower() == "unknown date":
            return date_str
        
        from datetime import datetime
        datetime_obj=datetime.strptime(date_str, "%m/%d/%Y").date()
        new_date_str = datetime_obj.strftime("%Y-%m-%d")
        
        return new_date_str

    def chunk_text_by_date(self):
        self.text=self.text.replace("EMPI,EPIC_PMRN,MRN_Type,MRN,Report_Number,Report_Date_Time,Report_Description,Report_Status,Report_Type,Report_Text",'')
        # chunks=selected_text.split('[report_end]')
        # chunks=[chunk for chunk in chunks if chunk!='']
        # dates=[ for chunk in chunks]
        
        self.chunks=[]
        self.chunk_metadata = []
        prev_length=0
        for note in self.text.split('[report_end]'):

            if note=='':
                continue
            # Date: split the 1st 100 characters by the header separator ','
            # (100 is arbitrarily chosen so that less splitting has to be done)
            # then date and time are the 5th element, with a space separating 
            # date and time.  
            date=self.reformat_date(note[:100].split(',')[5].split(' ')[0])
            length=sum([self.count_tokens_in_word(word) for word in note.split()])

            if self.chunks!=[] and (date==self.chunk_metadata[-1]['date'] and (length+prev_length)<self.token_limit):
                # If the current and previous notes are from the same day, combine them,
                # unless the combined chunk would exceed self.token_limit.
                self.chunks[-1] += note

            else:
                self.chunks.append(note)

                self.chunk_metadata.append({
                    'date': date,
                    'all_dates': [date],
                    'date_range': date
                    })

            prev_length=length


    def chunk_text_by_tokenlimit(self):
        """
        Splits the text into smaller segments based on the token limit.
        Uses clinical date extraction to power temporal tracking.
        """
        import re
        
        # Line-by-line approach to track dates and void birthdays
        lines = self.text.split('\n')
        if self.debug:
            print(f"Total characters to process: {len(self.text)}")
            
        self.chunks = []
        self.chunk_metadata = []
        current_chunk = []
        current_chunk_tokens = 0
        current_date = "Unknown Date"
        dates_in_chunk = []

        words = self.text.split()
        for i, word in enumerate(words):
            # Look for a date pattern in the word
            date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', word)
            if date_match:
                potential_date = date_match.group(1)
                # Check a window of surrounding words to exclude DOB
                window = " ".join(words[max(0, i-5):i+6]).upper()
                if not any(x in window for x in ["DOB", "BIRTH", "BORN", "B.DAY"]):
                    if potential_date != current_date:
                        current_date = potential_date
                        marker_str = f"[REPORT DATE: {current_date}] "
                        current_chunk.append(marker_str)
                        current_chunk_tokens += len(marker_str.split())
                        if current_date not in dates_in_chunk:
                            dates_in_chunk.append(current_date)

            
            tokens_in_word = self.count_tokens_in_word(word)
            
            if current_chunk_tokens + tokens_in_word <= self.token_limit:
                current_chunk.append(word)
                current_chunk_tokens += tokens_in_word
            else:
                self.chunks.append(' '.join(current_chunk))
                
                # Store extensive date metadata for transparency
                date_range = f"{dates_in_chunk[0]} to {dates_in_chunk[-1]}" if dates_in_chunk else "Unknown Date"
                self.chunk_metadata.append({
                    'date': current_date, # Legacy single-date fallback
                    'all_dates': list(dates_in_chunk),
                    'date_range': date_range
                })
                
                context_str = f"[CONTINUED FROM REPORT DATE: {current_date}]"
                current_chunk = [context_str, word]
                current_chunk_tokens = len(context_str.split()) + 1 + tokens_in_word
                dates_in_chunk = [current_date] if current_date != "Unknown Date" else []

        if current_chunk:
            self.chunks.append(' '.join(current_chunk))
            date_range = f"{dates_in_chunk[0]} to {dates_in_chunk[-1]}" if dates_in_chunk else "Unknown Date"
            self.chunk_metadata.append({
                'date': current_date,
                'all_dates': list(dates_in_chunk),
                'date_range': date_range
            })
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
    
    