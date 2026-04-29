from doctest import debug
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

class PostProcessing:
    '''
    A class for post-processing operations on CSV files containing abstracts.
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
        print(f"Found {self.concatenated_df['OpenAI_Screen_Abstract'].sum()} abstracts")
        print(f"Saved Master List to: \n {self.final_file_path}")
    
    @staticmethod
    def add_raw_results_to_master_list(master_list_path, raw_results_path, debug=False, filename_col='PMID'):
        '''Merges results into the master list, assuming the files have names which are stored in a column in the master list.'''
        master_df = pd.read_csv(master_list_path)
        raw_results_df = pd.read_csv(raw_results_path)

        new_columns = [col for col in raw_results_df.columns if col != raw_results_df.columns[0]]
        for column in new_columns:
            if column[:11].upper() == 'EXPLANATION' or column[:5].upper() == 'CHUNK' or 'dates' in column.lower():
                default_value = ''
            else: 
                default_value = np.nan
            if column not in master_df.columns:
                master_df[column] = default_value

        for _, row in tqdm(raw_results_df.iterrows(), desc='Updating master list',total=len(raw_results_df)):
            filename = str(row.iloc[0]).removesuffix('_OCR').lstrip('0')
            # Match by stripping leading zeros from both the master list MRNs and the results file MRNs
            matching_index = master_df[master_df[filename_col].astype(str).str.lstrip('0') == filename].index
            if len(matching_index) > 0:
                for column in new_columns:
                    master_df.loc[matching_index, column] = row[column]
        if not debug:
            master_df.to_csv(master_list_path, index=False)
        return master_df
    
    @staticmethod
    def update_emr_master_list(master_list_path, raw_results_path, debug=False, matching_col='MRN'):
        '''Merges results into the master list, based on MRN'''
        master_df = pd.read_csv(master_list_path)
        raw_results_df = pd.read_csv(raw_results_path)

        master_df = pd.merge(master_df, raw_results_df, on=matching_col, how='outer')
        if not debug:
            master_df.to_csv(master_list_path, index=False)
        return master_df

    @staticmethod
    def sort_master_list(df, questions_dict):
        
        new_col_order=['MRN','filepath']
        
        inc_qs=sorted([colname for colname in df.columns if colname.startswith('inclusion_')])
        inc_expls=sorted([colname for colname in df.columns if colname.startswith('EXPLANATION: inclusion_')])

        new_col_order+=[val for pair in zip(inc_qs, inc_expls) for val in pair]
        
        for question in questions_dict.values():

            colnames=[question]
            explanations=sorted([col for col in df.columns if f'_explanations: {question}' in col])
            new_col_order+=colnames+explanations

            onset=f'onset_date: {question}'
            if onset in df.columns:
                new_col_order.append(onset)
        
        stuff_that_shouldnt_be_here=[col for col in df.columns if col not in new_col_order]
        new_col_order+=stuff_that_shouldnt_be_here
        
        return df[new_col_order]

            
    @staticmethod
    def finalize_master_list(master_list_path, questions_dict):

        master_df = pd.read_csv(master_list_path)
        
        new_cols=master_df.columns.values.tolist()
        short_qs_counts={}
        new_questions_dict=questions_dict
        for long_q, short_q in questions_dict.items():
            if short_q in short_qs_counts.keys(): #handles repeat names
                short_qs_counts[short_q]+=1
                short_q=f"{short_q}_{short_qs_counts[short_q]}"
                new_questions_dict[long_q]=short_q
            else:
                short_qs_counts[short_q]=1
            new_cols=[col.replace(long_q, short_q) for col in new_cols]

        master_df.columns = new_cols
        master_df=PostProcessing.sort_master_list(master_df, new_questions_dict)
        master_df.to_csv(master_list_path, index=False)
        
        q_key=pd.DataFrame()
        q_key['question_name']=questions_dict.values()
        q_key['question_text']=questions_dict.keys()

        questions_key_path='/'.join(master_list_path.split('/')[:-1]+['question_key.csv'])
        if os.path.isfile(questions_key_path):
            old_q_key=pd.read_csv(questions_key_path)
            q_key=pd.concat([old_q_key, q_key])

        q_key.to_csv(questions_key_path, index=False)


        return master_df

    def clean_up(self):
        """Method to remove just the specific preprocessing CSVs"""
        removal_list = ['_cleaned', '_filtered']
        directory = os.path.dirname(self.final_file_path)
        for filename in os.listdir(directory): 
            if any(removal_tag in filename for removal_tag in removal_list):
                try: os.remove(os.path.join(directory, filename))
                except: pass
    
    def run(self):
        self.merge_csvs_on_abstract()
        self.concatenate_csvs()
        self.clean_up()
        return self.concatenated_df
    