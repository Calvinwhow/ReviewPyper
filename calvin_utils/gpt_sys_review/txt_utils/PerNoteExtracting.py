import pandas as pd
import os
from tqdm import tqdm
from math import isnan
from calvin_utils.gpt_sys_review.txt_utils.ClinicalNotesExtractor import ClinicalNotesExtractor
import json
from datetime import datetime

class PerNoteExtracting(ClinicalNotesExtractor):
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
    def __init__(self, input_file_list, mrn_file, output_dir, filter_list=None, separator="|", MRN_str='MRN', report_end_str='[report_end]',debug=False):

        super().__init__(input_file_list, mrn_file, output_dir, filter_list=None, separator="|", MRN_str='MRN', report_end_str='[report_end]',debug=False)
    

    ### Public API ###
    @staticmethod
    def append_json(filepath, mrn, dictionary, write_new):
        if write_new:
            with open(filepath, 'w', encoding="UTF-8") as file:
                json.dump(dictionary, file)
        else:
            with open (filepath, mode="r+", encoding="UTF-8") as file:
                file.seek(os.stat(filepath).st_size -1)
                file.write( f",{json.dumps(dictionary)[1:]}")  


    def split_notes(self, file, make_master_file=False):
        """
        Splits the file so that each subject is in their own file, 
        explicitly sorting all reports chronologically.
        """
        
        reader=self._file_reader(file)
        # Identify the MRN column index from the header row
        
        for row in reader:
            self.header_description=row
            break
        self._get_header_info()

        note=''
        header = ''
        skipped_mrns=[]
        all_headers=[] # used to stop duplicate notes from being added to a subject's file 
        file_list={}

        for row in reader:
            
            note+=row

            if self.separator in row:
                header=row

            if (self.report_end_str in row) or (row == ''):
                
                if header=='':
                    raise ValueError("Header is empty")
                
                elif header in all_headers:
                    if self.debug:
                        print(f"Warning: Duplicate note with header '{header.strip()}' found in {file}. Skipping this note to avoid duplicates in the output.")
                    note=''
                    header=''
                    continue
                
                else:
                    all_headers.append(header)

                split_header=header.split(self.separator)
                note_mrn=split_header[self.mrn_index]
                report_id=split_header[self.report_id_index]
                mrn = self._map_mrn(note_mrn)
                
                note_date=datetime.strptime(split_header[self.date_index].split(' ')[0], '%m/%d/%Y')
                
                if self.selected_mrns and int(mrn) not in self.selected_mrns:
                    pass

                elif mrn is False and note_mrn not in skipped_mrns:  #only print the warning the first time we encounter a given unmapped MRN, but skip all notes with that MRN
                    skipped_mrns.append(note_mrn)
                    print(f"Warning: MRN {note_mrn} from {file} not found in {self.mrn_file}. Skipping note")

                elif mrn is not False:
                    path=os.path.join(self.raw_files_dir, f'{mrn}.json')
                    dict_to_write={report_id:{'date':note_date.date().strftime('%m/%d/%Y'),'text':note}}
                    make_new_file=(False if mrn in list(file_list.keys()) else True)

                    self.append_json(path,mrn,dict_to_write, write_new=make_new_file)

                    file_list[mrn]=path

                note=''
                header=''

        return file_list

    def sort_and_write_to_text(self, filelist):
        
        for filepath in filelist:
            
            with open(filepath, 'r', encoding="UTF-8") as file:
                subject_dict=json.load(file)
                sorted_dict=dict(sorted(subject_dict.items()))

            with open(filepath.replace('.json','.txt'),'w', encoding="UTF-8") as text_file:
                text_file.write(self.header_description)

                for report_number, data in sorted_dict.items():
                    text_file.write(data['text'])

    def run(self, selected_mrns=None, make_master_file=False):
        """ Runs the entire process of splitting the files, generating the master list, and filtering it."""
        
        file_list=self.split_notes(self.combine_files_temp(), make_master_file=make_master_file)

        # for file in self.input_file_list:
        #     self.split_by_subject(file)
        self.sort_and_write_to_text(file_list.values()) 
        self.generate_master_list()
        # self.filter_master_list(selected_mrns)
        self.save_master_list()
        return self.master_list
    
