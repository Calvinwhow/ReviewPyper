import pandas as pd

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