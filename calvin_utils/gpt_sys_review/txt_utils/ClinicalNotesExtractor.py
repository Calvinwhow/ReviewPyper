import pandas as pd
import os

class ClinicalNotesExtractor:
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
    def __init__(self, input_file_list, mrn_file, output_dir, filter_list=None, separator="|", MRN_str='MRN', report_end_str='[report_end]'):
        self.MRN_str = MRN_str
        self.report_end_str=report_end_str
        self.separator=separator
        self.input_file_list=input_file_list
        self.mrn_file=mrn_file

        self.mrn_mappings = self._prep_mrn_mapping(self.mrn_file)
        self.output_dir=output_dir
        self.raw_files_dir = output_dir + '_separated'

        if filter_list is None:
            self.filter_mrns=False
        else:
            self.selected_mrns=list(set([ self._map_mrn(mrn) for mrn in filter_list if self._map_mrn(mrn) is not False]))
            self.filter_mrns=True

        self._prep_out_dir()
    
    ### Internal API ###

    def _prep_out_dir(self):
        if not os.path.exists(self.raw_files_dir):
            os.makedirs(self.raw_files_dir, exist_ok=True)

    def _file_reader(self,file):
        """Generator to read the input file line by line."""
        for row in open(file, "r", encoding='utf-8'):
            yield row

    def _prep_mrn_mapping(self, mapping_file):
        mrn_mappings={}
        with open(mapping_file, 'r', encoding='utf-8') as f:

            for line in f:
                
                if line.startswith('IncomingId') or line.strip() == '':
                    # print("Skipping header or empty line in MRN file.")
                    continue
                
                parts = line.split('|') # This '|' is NOT related to self.separator; it is used in the MRN files from RPDR.
                primary_mrn = parts[0]

                # Note: I want the keys to be int (no leading zero issues) 
                # and the values to be str (keep leading zeroes, for recordkeeping).
                # May be tricky. self._map_mrn converts to int for this reason. 
                mrn_mappings[int(primary_mrn)] = primary_mrn 
                mrn_mappings.update({int(mrn): primary_mrn for mrn in parts[4:-1] if mrn.strip() != ''})

        return mrn_mappings

    def _map_mrn(self,mrn):
        
        if int(mrn) not in self.mrn_mappings.keys():
            return False
        else: 
            return self.mrn_mappings[int(mrn)]


    ### Public API ###

    def split_by_subject(self, file):
        """Splits the file so that each subject is in their own file."""
        reader=self._file_reader(file)
        reader=self._file_reader(file)
        
        for row in reader:
            file_header=row
            mrn_index=file_header.split(self.separator).index(self.MRN_str)
            break
        
        note=''

        for row in reader:
            
            note+=row

            if self.separator in row:
                header=row

            if (self.report_end_str in row) or (row == ''):

                if header=='':
                    raise ValueError("Header is empty")
                
                note_mrn=header.split(self.separator)[mrn_index]
                
                subject_mrn=self._map_mrn(note_mrn)

                if subject_mrn is False:
                    print(f'Warning: MRN {note_mrn} from {file} not found in {self.mrn_file}. Skipping note.')

                elif (self.filter_mrns is False) or (subject_mrn in self.selected_mrns):
                    with open(os.path.join(self.raw_files_dir, f'{subject_mrn}.txt'), 'a', encoding='utf-8') as subject_file:
                        subject_file.write(note)

                note=''
                header=''
    
    def generate_master_list(self):
        """Generates a master list of all subjects and their notes."""
        
        master_list = []
        for filename in os.listdir(self.raw_files_dir):
            if filename.endswith('.txt'):
                mrn = filename.split('.')[0]
                filepath = os.path.join(self.raw_files_dir, filename)
                master_list.append({'MRN': mrn, 'filepath': filepath})
        
        self.master_list = pd.DataFrame(master_list)
        
    def filter_master_list(self, selected_mrns=None):
        """Filters the master list based on selected MRNs."""
        
        if selected_mrns:
            selected_mrns_int = [int(mrn) for mrn in selected_mrns]
            self.master_list = self.master_list[self.master_list['MRN'].astype(int).isin(selected_mrns_int)]
        else:
            print("No list of MRNs given, keeping all subjects in the master list.")
        

    def save_master_list(self):
        """Saves the master list to a CSV file."""

        master_list_path = os.path.join(self.output_dir, 'master_list.csv')
        self.master_list.to_csv(master_list_path, index=False)
        print(f"Master list saved to {master_list_path}")


    def run(self, selected_mrns=None):
        """ Runs the entire process of splitting the files, generating the master list, and filtering it."""
        for file in self.input_file_list:
            self.split_by_subject(file) 
        self.generate_master_list()
        self.filter_master_list(selected_mrns)
        self.save_master_list()
        return self.master_list
    
