import os
import re
import pandas as pd

# Updated TextPreprocessor class without removing newlines
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
        self.output_dir = os.path.join(input_dir, '..', 'preprocessed')

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

    def process_files(self):
        """
        Reads each text file from the input directory, applies preprocessing, and saves it to the output directory.
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Loop through each file in the input directory
        for filename in os.listdir(self.input_dir):
            if filename.endswith('.txt'):
                input_filepath = os.path.join(self.input_dir, filename)
                
                output_filepath = os.path.join(self.output_dir, filename) #<-- edit
                
                # Read the original text
                with open(input_filepath, 'r', encoding='utf-8') as input_file:
                    original_text = input_file.read()
                
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
            
            if self.debug:
                print(f"Final chunk added to chunks list.")
    
    def get_chunks(self):
        """
        Returns the list of generated text chunks.
        
        Returns:
        - list: List containing the generated text chunks.
        """
        return self.chunks

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
        
    def separate_abstracts(self):
        """Separate the content into individual abstracts based on the described pattern."""
        #r'(\d+\.\s)'
        first_abstract_entry = re.search(r'(\d+\.\s[A-Z])', self.content)
        first_start_position = first_abstract_entry.start() if first_abstract_entry else None

        #r'\n(\d+\.\s)'
        # Find subsequent abstracts using the original pattern
        abstract_entries = re.finditer(r'\n\n\n(\d+\.\s[A-Z])', self.content)
        start_positions = [match.start() for match in abstract_entries]
        
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
        # Filter the abstracts dataframe using the mask indices
        self.filtered_df = self.abstracts_df.iloc[mask_indices]

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
    
class PostProcessing:
    '''
    A class for post-processing operations on CSV files containing abstracts.
    
    Example usage:
    post_process = PostProcessing("csv1.csv", "csv2.csv", "pubmed_csv.csv")
    merged_df = post_process.merge_csvs_on_abstract()
    final_file_path = post_process.concatenate_csvs()
    '''
    def __init__(self, file1_path, file2_path, pubmed_csv_path):
        '''
        Initialize the PostProcessing class.
        
        Parameters:
        file1_path (str): The path to the first CSV file to be merged.
        file2_path (str): The path to the second CSV file to be merged.
        pubmed_csv_path (str): The path to the PubMed CSV file for concatenation.
        '''
        self.file1_path = file1_path
        self.file2_path = file2_path
        self.pubmed_csv_path = pubmed_csv_path
        self.merged_df = None

    def merge_csvs_on_abstract(self):
        '''
        Merge two CSV files based on the 'Abstract' column and store the result.
        
        Returns:
        DataFrame: The merged DataFrame.
        '''
        # Reading the two CSV files
        df1 = pd.read_csv(self.file1_path)
        df2 = pd.read_csv(self.file2_path)
        
        # Merging them on the 'Abstract' column
        self.merged_df = pd.merge(df1, df2, on='Abstract', how='outer')
        
        return self.merged_df

    def concatenate_csvs(self):
        '''
        Concatenate the merged DataFrame with another CSV file and save as "master_list.csv".
        
        Returns:
        str: The path to the saved final CSV file.
        '''
        # Reading the PubMed CSV file
        df_pubmed = pd.read_csv(self.pubmed_csv_path)
        
        # Concatenating the dataframes
        self.concatenated_df = df_pubmed.join(self.merged_df, lsuffix='_merged', rsuffix='_pubmed')
        
        # Saving the concatenated DataFrame
        self.final_file_path = os.path.join(os.path.dirname(self.file1_path), "master_list.csv")
        self.concatenated_df.to_csv(self.final_file_path, index=False)
        print(f"Saved Master List to: \n {self.final_file_path}")
            
    def run(self):
        self.merge_csvs_on_abstract()
        self.concatenate_csvs()
        return self.concatenated_df