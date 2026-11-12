import pandas as pd
import json
import shutil
import os
from datetime import datetime
from pydantic.utils import deep_update

class BIDSOrganizer():
    
    def __init__(self, bids_root, data_dir, overwrite_questions=False,overwrite_preprocessed_files=True):
        self.bids_root = bids_root
        self.data_dir = data_dir
        self.overwrite_questions = overwrite_questions
        self.overwrite_preprocessed_files = overwrite_preprocessed_files
        self.chunks_dir=os.path.join(self.data_dir, 'json_chunks')
        if not os.path.exists(self.chunks_dir):
            print(f"No 'json_chunks' folder found in {self.data_dir}. Not including chunks in BIDS database")
            self.chunks_dir=False

        self.env=self.load_env()
        self.master_list = self.load_master_list()
        self.new_metadata=self.prep_new_metadata()
        self.create_bids_structure()

    @staticmethod
    def find_latest_filename(filename_list, number_getter, initial_name=False):

        if len(filename_list) == 1:
            return filename_list[0]
        else:
            if initial_name:
                filename_list.remove(initial_name) # can't be the default if there are other lists, and lack of number screws up next step
            numbered={pd.to_numeric(number_getter(name), errors='coerce'): name for name in filename_list}
            return numbered[max(numbered.keys())]
           
    @staticmethod
    def read_json(json_file_path):
        """Reads JSON data from a file and returns it as a dictionary."""
        try:
            with open(json_file_path, 'r', encoding='UTF-8') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Error: File {json_file_path} not found.")
            return {}
        # except json.JSONDecodeError:
        #     print("Error: Could not decode the JSON file.")
        #     return {}

    @staticmethod
    def save_to_json(output_path,dict):

        with open(output_path, 'w') as out_f:
            json.dump(dict, out_f, indent=4)


    def load_master_list(self):

        lists = [file for file in os.listdir(self.data_dir) if 'master_list' in file]
        if lists==[]:
            raise FileNotFoundError("No master_list CSV file found in the data directory.")
        
        elif len(lists) == 1:
            self.master_list_name =  lists[0]
        
        else:
            lists.remove('master_list.csv') # can't be the default if there are other lists, and lack of number screws up next step
            numbered={pd.to_numeric(name.split('_')[-1].replace('.csv', ''), errors='coerce'): name for name in lists}
            max_list_file=numbered[max(numbered.keys())]
            self.master_list_name = max_list_file

        return pd.read_csv(os.path.join(self.data_dir,self.master_list_name), dtype={'MRN': str})

    def load_env(self):

        env_list=[name for name in os.listdir(self.data_dir) if 'env' in name]
        if env_list==[]:
            print(f'Could not find env in {self.data_dir}. Skipping')
            return None
    
        env_name=self.find_latest_filename(env_list, initial_name='env.json',
                                            number_getter=lambda name: name.split('_')[-1].replace('.json',''))
        env=self.read_json(os.path.join(self.data_dir, env_name))
            # selected_mrns needs to be removed for PII reasons. 
            # We don't actually use it here, but this step is still necessary
        selected_mrns=env.pop("select_mrns", None)

        return env
    
    def create_bids_structure(self):
        # Create the BIDS directory structure
        os.makedirs(self.bids_root, exist_ok=True)
        subject_dirs={}

        for subject in self.master_list['MRN'].unique():
            # Should probably edit this to take preexisting folder names and not just the MRNS used by reviewpyper
            subject_dir = os.path.join(self.bids_root, f'sub-{subject}')
            subject_dirs[subject] = subject_dir
            # os.makedirs(subject_dir, exist_ok=True)
            os.makedirs(os.path.join(subject_dir, 'ReviewPyper'), exist_ok=True)
        
        self.master_list['bids_directory'] = subject_dirs.values()
        self.bids_folders = {subject: os.path.join(dir, 'ReviewPyper') for subject, dir in subject_dirs.items()}


    def move_txt_files_to_bids(self):

        for mrn in self.master_list['MRN'].unique():

            to_move ={'separated': os.path.join(self.data_dir, '_separated', f"{mrn}.txt"),
                      'preprocessed': os.path.join(self.data_dir, '_preprocessed', f"{mrn}.txt"),}
            
            for name, source_file in to_move.items():

                dest=os.path.join(self.bids_folders[mrn], f"{mrn}_{name}.txt")
                
                if not os.path.exists(source_file):
                    print(f"File {source_file} does not exist for subject {mrn}.")
                elif not self.overwrite_preprocessed_files and os.path.exists(dest):
                    continue
                else:
                    shutil.copy(source_file, dest)


    # def move_chunks_to_bids(self):

    #     if not os.path.exists(os.path.join(self.data_dir, 'json_chunks')):
    #         print(f"No '_chunks' directory found in {self.data_dir}.")
    #         return
        
    #     all_chunks = os.listdir(os.path.join(self.data_dir, 'json_chunks'))
    #     for chunk in all_chunks:
        
    #         chunk_mrn=chunk.split('_')[0]

    #         if chunk_mrn in self.master_list['MRN'].values:
    #             dest=os.path.join(self.bids_folders[chunk_mrn], chunk)
    #             shutil.copy(os.path.join(self.data_dir, 'json_chunks', chunk), dest)
    #         else:
    #             print(f"Chunk {chunk} does not correspond to any MRN in the master list.")


    def get_subject_chunks(self, mrn, run_suffix=''):

        if self.chunks_dir is False:
            return {}
        
        chunk_files = [file for file in os.listdir(self.chunks_dir) if mrn in file]
        chunk_json = self.find_latest_filename(chunk_files, 
                                               lambda x:x.split('_')[-1].replace('.json',''), 
                                               f"{mrn}_chunks.json")
        chunks_dict=self.read_json(os.path.join(self.chunks_dir,chunk_json))
        if run_suffix=='':
            return chunks_dict
        
        renamed_dict={chunk_name:{} for chunk_name in chunks_dict.keys()}

        for chunk_name, chunk_data in chunks_dict.items():
            renamed_dict[chunk_name]['text'+run_suffix]=chunk_data['text']
            renamed_dict[chunk_name]['metadata'+run_suffix]=chunk_data['metadata']
        return renamed_dict 


    def build_json_structure(self, mrn, subject_dict, question_dict, run_suffix=''):
        
        new_dict={}
        new_dict['questions']={q_name:{'question_text'+run_suffix:q_text} for q_name, q_text in question_dict.items()}

        for q_name, q_data in new_dict['questions'].items():

            full_q_text=q_data['question_text'+run_suffix]
            new_dict['questions'][q_name]['answers'+run_suffix]=subject_dict[full_q_text]
            new_dict['questions'][q_name]['explanations'+run_suffix]=subject_dict.get(f"EXPLANATION: {full_q_text}", '')

        new_dict['chunks']=self.get_subject_chunks(mrn, run_suffix)
        new_dict['metadata'+run_suffix]=subject_dict['metadata']
        return new_dict


    def merge_json(self, old_json_path, new_data,):

        if os.path.isfile(old_json_path):
            
            old_data=self.read_json(old_json_path)
            # old_data.update(new_data)
            merged=deep_update(old_data, new_data)
            self.save_to_json(old_json_path, merged)

        else:
            self.save_to_json(old_json_path,new_data)

            
    def prep_new_metadata(self,):
        new_metadata={}
        new_metadata['date_uploaded']=datetime.now().strftime('%Y/%M/%D %H:%M:%S')
        new_metadata['original_directory']=self.data_dir
        new_metadata['run_env'] = self.env

        return new_metadata


    def update_bids_jsons(self, json_file_path):

        extraction_json = self.read_json(json_file_path)
        json_name = os.path.basename(json_file_path)

        for mrn, data in extraction_json.items():

            if mrn not in self.master_list['MRN'].values:
                print(f"MRN {mrn} from the JSON does not correspond to any MRN in the master list.")
                continue

            output_file_path = os.path.join(self.bids_folders[mrn], f"{mrn}_{json_name}")

            if os.path.isfile(output_file_path):

                old_data=self.read_json(output_file_path)
                metadata_keys = [key for key in old_data.keys()  if 'metadata' in key]
                latest_key=self.find_latest_filename(metadata_keys,
                                                     lambda x: x.split('_')[-1])
                run_number=int(latest_key.split('_')[-1])+1
                
            else: 
                run_number=1
    
            question_dict={q_name:q_text for q_text, q_name in self.env["extraction_questions"].items()}

            # original metadata is just chunk dates, which have been moved to the 'chunks' header, so not needed
            data['metadata']=self.new_metadata 

            rebuilt_data=self.build_json_structure(mrn, data, question_dict, run_suffix=f'_{run_number:03}')

            self.merge_json(output_file_path,rebuilt_data)


    def organize(self, json_file_path=None):

        if json_file_path is None:
            json_file_path=os.path.join(self.data_dir, 'json_evaluated/emr_strict_extraction_evaluations_mapped.json')

        self.move_txt_files_to_bids()
        # self.move_chunks_to_bids()
        self.update_bids_jsons(json_file_path)
        self.master_list.to_csv(os.path.join(self.bids_root, self.master_list_name), index=False)

        print('Moved all files to BIDS directories at '+self.bids_root)

