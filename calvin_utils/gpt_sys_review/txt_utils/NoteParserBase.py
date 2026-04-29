import pandas as pd
import os
from tqdm import tqdm
import json
from datetime import datetime

class NoteParserBase():
    """
    A class to process clinical notes from an RPDR request and organize them in a csv.

    Attributes:
    - input_file_list (list): List of paths to all the text files you want to preproces.
    - output_dir (str): The directory where the output csv will be saved.

    Methods:
    - split_by_subject: splits the input file so that each subject has their own file.
    - generate_master_list: generates a master list of all subjects and the corresponding file.
    - filter_master_list: filters the master list based on a list of selected MRNs.
    - save_master_list: saves the master list to a csv file.
    - run: runs the entire process of splitting the files, generating the master list, and filtering it.
    """
    def __init__(self, separator="|",report_end_str='[report_end]',debug=False):

        self.report_end_str=report_end_str
        self.separator=separator
        self.debug=debug

    def _file_reader(self,file):
        """Generator to read the input file line by line."""
        for row in open(file, "r", encoding='utf-8'):
            yield row

    def _get_header_info(self, header_description):

        self.header_description=header_description
       
        split_description=self.header_description.split(self.separator)
        
        self.mrn_index=split_description.index(self.MRN_str)
        self.date_index=split_description.index('Report_Date_Time')
        self.report_id_index=split_description.index('Report_Number')
   

    ### Public API ###
    def split_text(self, file):
        
        reader=self._file_reader(file)
        # Identify the MRN column index from the header row
        for row in reader:
            self._get_header_info(row)
            break

        note=''
        header = ''
        date_list={}

        for row in reader:
            
            note+=row

            if self.separator in row:
                header=row

            if (self.report_end_str in row) or (row == ''):
                
                if header=='':
                    raise ValueError("Header is empty")
                
                split_header=header.split(self.separator)
                note_date=split_header[self.date_index].split(' ')[0]
                
                if note_date in date_list:
                    date_list[note_date]+='\n'+note
                else:
                    date_list[note_date]=note

                # path=os.path.join(self.raw_files_dir, f'{mrn}.json')
                # dict_to_write={report_id:{'date':note_date.date().strftime('%m/%d/%Y'),'text':note}}
                # self.append_json(path,mrn,dict_to_write, write_new=True if mrn in file_list else False)

                #     file_list[mrn]=path

        return date_list