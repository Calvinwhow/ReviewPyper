import re
import pandas as pd

class AbstractSeparator:
    '''
    A class to organize .txt of Abstracts from PubMed into a CSV file.
    
    Example usage:
    separator = AbstractSeparator("/path/to/your/textfile.txt")
    separator.separate_abstracts()
    separator.to_csv("/path/to/save/csvfile.csv")
    '''
    def __init__(self, file_path):
        self.file_path = file_path
        with open(self.file_path, 'r') as file:
            self.content = file.read()
        self.abstracts = []
    
    def identify_abstracts(self, reg_style ='flexbile'):
        """Function to use regex to identify abstracts"""
        if reg_style == 'rigid':
            r_string = '(\d+\.\s[A-Z])'
        elif reg_style == 'flexible':
            r_string = '(\d+\.\s[A-Za-z])'
        else:
            r_string = '(d+\.\s)'
            
        first_abstract_entry = re.search(r'(\d+\.\s[A-Za-z])', self.content)
        first_start_position = first_abstract_entry.start() if first_abstract_entry else None

        #r'\n(\d+\.\s)'
        # Find subsequent abstracts using the original pattern
        abstract_entries = re.finditer(r'\n\n\n(\d+\.\s[A-Za-z])', self.content)
        start_positions = [match.start() for match in abstract_entries]
        return first_start_position, start_positions

    def separate_abstracts(self):
        """Separate the content into individual abstracts based on the described pattern."""
        first_start_position, start_positions = self.identify_abstracts()
        
        if first_start_position is not None:
            start_positions = [first_start_position] + start_positions

        # Create abstract chunks based on the start positions
        abstract_chunks = [self.content[start_positions[i]:start_positions[i + 1]].strip() for i in range(len(start_positions) - 1)]
        abstract_chunks.append(self.content[start_positions[-1]:].strip())
        
        self.abstracts = abstract_chunks
    
    def to_csv(self, output_path=None):
        """Save the separated abstracts to a CSV."""
        df = pd.DataFrame(self.abstracts, columns=["Abstract"])
        if output_path is None:
            output_path = self.file_path.split('.txt')[0]+'_cleaned.csv'
        df.to_csv(output_path, index=False)
        return df, output_path
        
    def get_abstracts(self):
        """Return the list of separated abstracts."""
        return self.abstracts
    
    def run(self):
        """Orchestrator method"""
        self.separate_abstracts()
        df, output_path = self.to_csv()
        return df, output_path