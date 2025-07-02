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
    
class TitleReviewFilter():
    """
    A class to filter abstracts based on title review results.

    Methods:
    - load_data: Loads the title review results and abstracts data.
    - filter_abstracts: Filters the abstracts based on a specified column from the title review results.
    - save_filtered_data: Saves the filtered abstracts to a specified path.
    - get_filtered_dataframe: Returns the filtered abstracts dataframe for visualization.
    """

    def __init__(self, title_review_path, abstracts_path, column_name):
        """
        Initializes the TitleReviewFilter class with paths to the title review results and abstracts CSVs.

        Parameters:
        - column_name (str): The column name in the title review results to use for filtering.
        - title_review_path (str): Path to the title review results CSV.
        - abstracts_path (str): Path to the abstracts CSV.
        """
        self.title_review_path = title_review_path
        self.abstracts_path = abstracts_path
        self.title_df, self.abstracts_df = self.load_data()
        self.column_name = column_name

    def load_data(self):
        """
        Loads the title review results and abstracts data from CSVs.

        Returns:
        - DataFrame, DataFrame: DataFrames containing the title review results and abstracts.
        """
        title_df = pd.read_csv(self.title_review_path)
        abstracts_df = pd.read_csv(self.abstracts_path)
        return title_df, abstracts_df

    def filter_abstracts(self):
        """
        Filters the abstracts based on a specified column from the title review results.
        """
        # Find the indices of the rows in title review results where the specified column has a value of 1
        mask_indices = self.title_df[self.title_df[self.column_name] == 1].index
        
        try:
            self.filtered_df = self.abstracts_df.iloc[mask_indices]
        except IndexError:
            # Some of the indices in mask_indices don't exist in abstracts_df
            valid_indices = [i for i in mask_indices if 0 <= i < len(self.abstracts_df)]
            self.filtered_df = self.abstracts_df.iloc[valid_indices]

    def save_filtered_data(self, output_path=None):
        """
        Saves the filtered abstracts to a specified path.

        Parameters:
        - output_path (str): Path to save the filtered abstracts CSV.
        """

        if output_path is None:
            output_path = self.abstracts_path.split('.')[0]+'_filtered.csv'
        if not output_path.endswith('.csv'):
            output_path += '.csv'
        self.filtered_df.to_csv(output_path, index=False)
        return output_path

    def get_filtered_dataframe(self):
        """
        Returns the filtered abstracts dataframe for visualization.

        Returns:
        - DataFrame: DataFrame containing the filtered abstracts.
        """
        return self.filtered_df
    
    def run(self):
        """
        Orchestrator method.
        """
        self.filter_abstracts()
        output_path = self.save_filtered_data()
        df = self.get_filtered_dataframe()
        return df, output_path