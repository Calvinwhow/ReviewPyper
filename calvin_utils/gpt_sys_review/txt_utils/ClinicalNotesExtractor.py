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
    def __init__(self, input_file_list, output_dir, separator="|", MRN_str='MRN', report_end_str='[report_end]'):
        self.MRN_str = MRN_str
        self.report_end_str=report_end_str
        self.separator=separator
        self.input_file_list=input_file_list
        self.output_dir=output_dir
        
        self._prep_out_dir()
    
    ### Internal API ###

    def _prep_out_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def _file_reader(self,file):
        """Generator to read the input file line by line."""
        for row in open(file, "r", encoding='utf-8'):
            yield row

    ### Public API ###

    def split_by_subject(self, file):
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
            file_header = row
            try:
                mrn_index = file_header.split(self.separator).index(self.MRN_str)
            except ValueError:
                continue 
            break
        
        # Dictionary to store reports per MRN: {mrn: [(date, full_note_text), ...]}
        subject_reports = defaultdict(list)
        
        note = ''
        header = ''

        for row in reader:
            note += row

            if self.separator in row:
                header = row

            if (self.report_end_str in row) or (row == ''):
                if header == '':
                    continue 
                
                parts = header.split(self.separator)
                if len(parts) > mrn_index:
                    mrn = parts[mrn_index]
                    
                    # Extract date for sorting
                    # Look for Encounter Date, Visit Date, or the |date time| pattern in RPDR headers
                    date_str = "01/01/1900"
                    # Pattern 1: Labels in the note body
                    date_match = re.search(r'(?:Encounter Date:|Visit Date:|Dated:|Signed:)\s*(\d{1,2}/\d{1,2}/\d{4})', note, re.IGNORECASE)
                    
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
            
            output_path = os.path.join(self.output_dir, f'{mrn}.txt')
            with open(output_path, 'w', encoding='utf-8') as subject_file:
                for _, report_text in reports:
                    subject_file.write(report_text)
                    subject_file.write("\n")

    
    def generate_master_list(self):
        """Generates a master list of all subjects and their notes."""
        
        master_list = []
        for filename in os.listdir(self.output_dir):
            if filename.endswith('.txt'):
                mrn = filename.split('.')[0]
                filepath = os.path.join(self.output_dir, filename)
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
    
