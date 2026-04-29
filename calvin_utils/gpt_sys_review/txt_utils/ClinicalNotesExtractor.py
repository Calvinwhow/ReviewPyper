import pandas as pd
import os
from tqdm import tqdm
from math import isnan

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
    def __init__(self, input_file_list, mrn_file, output_dir, filter_list=None, separator="|", MRN_str='MRN', report_end_str='[report_end]',debug=False):
        self.MRN_str = MRN_str
        self.report_end_str=report_end_str
        self.separator=separator
        self.input_file_list=input_file_list
        self.mrn_file=mrn_file
        self.output_dir=output_dir
        self.raw_files_dir = output_dir + '_separated'
        self.debug=debug

        if filter_list is not None:
            self.selected_mrns=[int(mrn) for mrn in filter_list]
        else:
            self.selected_mrns=None
        self._prep_out_dir()
        if mrn_file is not None:
            self.mrn_mappings = self._prep_mrn_mapping(self.mrn_file)

        if self.selected_mrns:
            invalid_selections=[[mrn,self._map_mrn(mrn)] for mrn in self.selected_mrns if int(mrn)!=int(self._map_mrn(mrn))]

            if len(invalid_selections)>0:
            
                df=pd.DataFrame(invalid_selections,columns=['input_mrn','equivalent_mrn'])
                if not df['equivalent_mrn'].all():
                    not_found=df[df['equivalent_mrn'].astype(str)=='False']

                    self.mrn_mappings.update({int(mrn):mrn for mrn in not_found['input_mrn']})
                    df=df[df['equivalent_mrn'].astype(str)!='False']
            
                df.to_csv(output_dir+'equivalent_mrns.csv', index=False)
                print('Some MRNs in selection list correspond to the same subject. Saving a list of equivalent MRNs')

    
    ### Internal API ###

    def _prep_out_dir(self):
        if not os.path.exists(self.raw_files_dir):
            os.makedirs(self.raw_files_dir, exist_ok=True)

    def _file_reader(self,file):
        """Generator to read the input file line by line."""
        for row in open(file, "r", encoding='utf-8'):
            yield row

    def _get_header_info(self, ):
       
        split_description=self.header_description.split(self.separator)
        self.mrn_index=split_description.index(self.MRN_str)
        self.date_index=split_description.index('Report_Date_Time')
        self.report_id_index=split_description.index('Report_Number')
   

    def _prep_mrn_mapping(self, mapping_file):
        df=pd.read_csv(mapping_file, sep='|', dtype=str)
        df.drop(columns=["IncomingSite","Status"], inplace=True)
        primaries=[]
        mrn_mappings={}

        for i, row in tqdm(df.iterrows(), desc='Prepping MRN mapping',total=len(df)):
            # print(row)
            primary=row['IncomingId']
            
            if self.selected_mrns is not None:

                is_selected=False
                for mrn in row.values:
                    if type(mrn)==float and isnan(mrn):
                        continue
                    elif int(mrn) in self.selected_mrns and (not pd.isna(mrn) and mrn.strip() != ''):
                        primary=mrn
                        is_selected=True
                        break

                if int(primary) in mrn_mappings.keys() and not is_selected:
                    primary=mrn_mappings[int(primary)] # if there are duplicates, use the first one we see in the file 

                elif int(primary) in mrn_mappings.keys() and is_selected:
                    # We now want to update the mapping to use the selected MRN as the primary.
                    # So, get the original primary, then update the old mappings
                    # that mapped to the original primary to now map to the selected primary. 
                    old_primary=mrn_mappings[int(primary)]
                    for mrn, mapped_primary in mrn_mappings.items():
                        if mapped_primary == old_primary:
                            mrn_mappings[int(mrn)] = primary

            mrn_mappings.update({int(mrn): primary for mrn in row if not pd.isna(mrn) and mrn.strip() != ''})

        if self.debug:
            map_df=pd.DataFrame(list(mrn_mappings.items()), columns=['MRN', 'Primary MRN'])
            map_df.to_csv(os.path.join(self.output_dir, 'mrn_mappings.csv'), index=False)
            print(f"Saved MRN mappings to {os.path.join(self.output_dir, 'mrn_mappings.csv')}")

        return mrn_mappings


    def _map_mrn(self,mrn):
        
        if int(mrn) not in self.mrn_mappings.keys():
            return False
        else: 
            return self.mrn_mappings[int(mrn)]


    ### Public API ###

    def split_by_subject(self, file, make_master_file=False):
        """
        Splits the file so that each subject is in their own file, 
        explicitly sorting all reports chronologically.
        """
        import re
        from datetime import datetime
        from collections import defaultdict

        reader = self._file_reader(file)
        
        # Identify the MRN column index from the header row
        for row in reader:
            self.header_description = row
            break
        self._get_header_info()


        # Dictionary to store reports per MRN: {mrn: [(date, full_note_text), ...]}
        subject_reports = defaultdict(list)
        
        note = ''
        header = ''
        skipped_mrns=[]
        all_headers=[] # used to stop duplicate notes from being added to a subject's file 
        
        for row in tqdm(reader,"Extracting notes",unit='lines'):
            note += row

            if self.separator in row:
                header = row

            if (self.report_end_str in row) or (row == ''):
                if header == '':
                    continue 
                elif header in all_headers:
                    if self.debug:
                        print(f"\nWarning: Duplicate note with header '{header.strip()}' found in {file}. Skipping this note to avoid duplicates in the output.")
                    continue
                else:
                    all_headers.append(header)
                
                parts = header.split(self.separator)
                note_mrn=parts[self.mrn_index]

                mrn = self._map_mrn(note_mrn)

                if self.selected_mrns and int(mrn) not in self.selected_mrns:
                    continue

                elif mrn is False and note_mrn not in skipped_mrns:
                    skipped_mrns.append(note_mrn)
                    print(f"\nWarning: MRN {note_mrn} from {file} not found in {self.mrn_file}. Skipping note")
                elif mrn is False:
                    continue #only print the warning the first time we encounter a given unmapped MRN, but skip all notes with that MRN

                elif len(parts) > self.mrn_index:
                    
                    # Extract date for sorting
                    # Look for Encounter Date, Visit Date, or the |date time| pattern in RPDR headers
                    date_str = "01/01/1900"
                    # Pattern 1: Labels in the note body
                    # date_match = re.search(r'(?:Encounter Date:|Visit Date:|Signed:)\s*(\d{1,2}/\d{1,2}/\d{4})', note, re.IGNORECASE)
                    
                    # Pattern 2: RPDR pipe-separated header date (e.g., |11/11/2016 3:30:00 PM|)
                    if not date_match:
                        date_match = re.search(r'\|\s*(\d{1,2}/\d{1,2}/\d{4})\s+\d{1,2}:\d{2}:\d{2}', note)

                    if date_match and "DOB" not in date_match.group(0):
                        # Use the last captured group as the date
                        date_str = date_match.groups()[-1]
                    elif len(parts) > 5:
                        # Fallback: Many RPDR notes have the date in the 6th | column (index 5)
                        header_date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', parts[5])
                        if header_date_match:
                            date_str = header_date_match.group(1)
                    
                    try: date_obj = datetime.strptime(date_str, "%m/%d/%Y")
                    except: date_obj = datetime(1900, 1, 1)

                    subject_reports[mrn].append((date_obj, note))

                note = ''
                header = ''

        # Sort and write each subject's file to guarantee chronological order
        print(f"Guaranteeing chronological order for {len(subject_reports)} subjects...")
        for mrn, reports in subject_reports.items():
            reports.sort(key=lambda x: x[0])
            
            output_path = os.path.join(self.raw_files_dir, f'{mrn}.txt')
            with open(output_path, 'w', encoding='utf-8') as subject_file:
                for _, report_text in reports:
                    subject_file.write(report_text)
                    subject_file.write("\n")

            if make_master_file:
                with open(self.output_dir+'all_selected_records.txt', 'a', encoding='utf-8') as master_file:
                    for _, report_text in reports:
                        master_file.write(report_text)
                        master_file.write("\n\n")

    
    def generate_master_list(self, file_ending='.txt'):
        """Generates a master list of all subjects and their notes."""
        
        master_list = []
        for filename in os.listdir(self.raw_files_dir):
            if filename.endswith(file_ending):
                mrn = filename.split('.')[0]
                filepath = os.path.join(self.raw_files_dir, filename)
                master_list.append({'MRN': mrn, 'filepath': filepath})
        
        self.master_list = pd.DataFrame(master_list)
        
    # def filter_master_list(self, selected_mrns=None):
    #     """Filters the master list based on selected MRNs."""
        
    #     if selected_mrns:
    #         selected_mrns_int = [int(mrn) for mrn in selected_mrns]
    #         self.master_list = self.master_list[self.master_list['MRN'].astype(int).isin(selected_mrns_int)]
    #     else:
    #         print("No list of MRNs given, keeping all subjects in the master list.")
        

    def save_master_list(self):
        """Saves the master list to a CSV file."""

        master_list_path = os.path.join(self.output_dir, 'master_list.csv')
        counter=0
        while os.path.isfile(master_list_path): #rename if master list already exists
            counter+=1
            master_list_path=master_list_path.replace('.csv',f"_{counter}.csv")
        self.master_list.to_csv(master_list_path, index=False)
        print(f"Master list saved to {master_list_path}")

    def combine_files_temp(self):
        new_input_file = os.path.join(self.output_dir, 'combined_files.txt')
        with open(new_input_file, 'w', encoding='UTF-8') as outfile:
            for fname in self.input_file_list:
                with open(fname, encoding='UTF-8') as infile:
                    for line in infile:
                        outfile.write(line)
        return new_input_file


    def run(self, selected_mrns=None, make_master_file=False):
        """ Runs the entire process of splitting the files, generating the master list, and filtering it."""
        self.split_by_subject(self.combine_files_temp(), make_master_file=make_master_file)
        # for file in self.input_file_list:
        #     self.split_by_subject(file) 
        self.generate_master_list()
        # self.filter_master_list(selected_mrns)
        self.save_master_list()
        return self.master_list
    
