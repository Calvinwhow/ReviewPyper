from tqdm import tqdm
from calvin_utils.gpt_sys_review.gpt_utils.openai_labeller import CaseReportLabeler
from calvin_utils.gpt_sys_review.gpt_utils.openai_summarizer import OpenAISummarizer
from fuzzywuzzy import fuzz
import pandas as pd
import numpy as np
import json
import os
import re
from fractions import Fraction

class SectionLabeler:
    """
    A class to label sections of a text document using LDA and OpenAI's GPT-3.

    Attributes:
    - folder_path (str): The path to the folder containing text files.
    - article_type (str): The type of article (e.g., 'research', 'case', 'emr').
    - lda_model (object): The trained LDA model for topic modeling.
    - vectorizer (object): The CountVectorizer object for text vectorization.
    - chunker (object): The TextChunker object for text chunking.
    - api_key (str): The OpenAI API key. [Replace with actual API key later]

    Methods:
    - train_lda: Trains the LDA model based on the text chunks.
    - dominant_topic: Finds the dominant topic for a given text chunk.
    - label_with_openai: Labels a section based on the dominant topic using OpenAI's GPT-3.
    - process_files: Processes all text files in the specified folder.
    - save_to_json: Saves the labeled sections to a JSON file.
    """

    def __init__(self, folder_path, article_type, api_key_path=None, is_azure=False, deployment_id=None, api_base=None, api_version=None):
        """
        Initializes the SectionLabeler class with the folder path and article type.

        Parameters:
        - api_key_path (str): Path to the file containing the OpenAI API key.
        - article_type (str): The type of article (e.g., 'research', 'case', 'emr').
        - folder_path (str): The path to the folder containing text files.
        - article_type (str): The type of article (e.g., 'research', 'case', 'emr').
        """
        self.api_key_path = api_key_path
        self.folder_path = folder_path
        self.article_type = article_type
        self.chunker = None
        self.output_dict = {}
        self.is_azure = is_azure
        self.deployment_id=deployment_id
        self.api_base=api_base
        self.api_version=api_version

    def select_labels(self):
        # Define section labels for each article type
        if self.article_type == "research":
            self.section_headers = {
            "Abstract": ["Abstract"],
            "Introduction": ["Background", "Introduction", "Intro"],
            "Methods": ["Methods", "Materials", "Material and Methods", "Materials & Methods", "Methodology", "Subjects and Methods"],
            "Results": ["Results", "Findings"],
            "Discussion": ["Discussion", "Interpretation"],
            "Conclusion": ["Conclusion", "Summary"],
            "References": ["References", "Bibliography", "Citations"]
            }
        elif self.article_type == "case":
            self.section_headers = {
            "Case_Report": ["yes", "y", "positive", "correct"]
            }
        elif self.article_type == "emr":
            self.section_headers = {
            "emr": ["yes", "y", "positive", "correct"]
            }
        elif self.article_type == "other":
            self.section_headers = {
                "Positive": ["yes", "y", "positive", "correct"]
            }
        else:
            raise ValueError(f"Unknown article type {self.article_type}, choose 'case', 'research', or 'emr'.")
            
    def get_questions(self, manual_question=None):
        if self.article_type == "research":
            questions = None
        elif self.article_type == "case":
            questions = {'Prioritizing implicit and explicit information, do you think this contains a description of a medical case? For example, if the text refers to a patient, seemingly describes a history of presenting illness, or is seemingly describing a medical situation. This could be in referring to hospital course, imaging findings, or laboratory results. ONLY RESPOND AS YES OR NO (Y/N)': 'case_report'}
        elif self.article_type == "emr":
            questions = {"Prioritizing implicit and explicit information, do you think this mentions anything about a medical patient? For example, if the text refers to a patient, seemingly describes a history of presenting illness, or is seemingly describing a medical situation. This could be in referring to hospital course, imaging findings, diagnoses, or laboratory results. ONLY RESPOND AS YES OR NO (Y/N)": 'emr'}
        elif self.article_type == "other":
            if manual_question is None:
                raise ValueError(f"Error, must enter question as a string into process_files. ex: (question='this is my question')")    
            else:
                questions = {manual_question : 'other'}
        else:
            questions = None
        return questions

    def label_with_exact_matching(self, text):
        labeled_sections = {}
        current_section = None
        current_text = ""

        # Adding newline to section labels for exact matching
        section_labels_with_newline = []
        for labels in self.section_headers.values():
            section_labels_with_newline.extend([f"\n{label}\n" for label in labels])

        for line in text.split('\n'):
            if f"\n{line}\n" in section_labels_with_newline:
                if current_section:
                    labeled_sections[current_section] = current_text.strip()
                current_section = line
                current_text = ""
            else:
                current_text += line + "\n"

        # Adding the last section
        if current_section:
            labeled_sections[current_section] = current_text.strip()

        return labeled_sections

    def label_with_fuzzy_matching(self, text, labeled_sections):
        current_section = None
        current_text = ""

        for line in text.split('\n'):
            for section, labels in self.section_headers.items():
                if any(fuzz.partial_ratio(line, label) > 80 for label in labels):
                    if current_section:
                        if current_section not in labeled_sections:
                            labeled_sections[current_section] = current_text.strip()
                        else:
                            labeled_sections[current_section] += "\n" + current_text.strip()
                    current_section = section
                    current_text = ""
                    break
            else:
                current_text += line + "\n"

        # Adding the last section
        if current_section:
            if current_section not in labeled_sections:
                labeled_sections[current_section] = current_text.strip()
            else:
                labeled_sections[current_section] += "\n" + current_text.strip()

        return labeled_sections

    def label_with_loose_matching(self, text, labeled_sections):
        current_section = None
        current_text = ""

        for line in text.split('\n'):
            for section, labels in self.section_headers.items():
                if any(re.search(f"\\b{label}\\b", line, re.IGNORECASE) for label in labels):
                    if current_section:
                        if current_section not in labeled_sections:
                            labeled_sections[current_section] = current_text.strip()
                        else:
                            labeled_sections[current_section] += "\n" + current_text.strip()
                    current_section = section
                    current_text = ""
                    break
            else:
                current_text += line + "\n"

        # Adding the last section
        if current_section:
            if current_section not in labeled_sections:
                labeled_sections[current_section] = current_text.strip()
            else:
                labeled_sections[current_section] += "\n" + current_text.strip()

        return labeled_sections

    def label_text(self, text, show_residuals=True):
        labeled_sections = self.label_with_exact_matching(text)
        labeled_sections = self.label_with_fuzzy_matching(text, labeled_sections)
        labeled_sections = self.label_with_loose_matching(text, labeled_sections)
        
        # Check for residual text
        labeled_text = "".join(list(labeled_sections.values()))
        residual_text = set(text) - set(labeled_text)
        if len(residual_text) > 100:
            print(f"Warning: High number of characters missed ({len(residual_text)}), please investigate manually.")
            if show_residuals:
                print(residual_text)
        return labeled_sections, residual_text
    
    def save_to_json(self, output_dict, filename=None):
        """
        Saves the labeled sections to a JSON file.

        Parameters:
        - output_dict (dict): Dictionary containing the labeled sections.

        Returns:
        - None
        """
        # Create a new directory in the same root folders
        root_dir = os.path.dirname(self.folder_path)
        out_dir = os.path.join(root_dir, 'json')
        os.makedirs(out_dir, exist_ok=True)
        
        if filename is not None and not os.path.exists(filename):
            save_file_path = os.path.join(out_dir, f'{filename}_labeled_sections.json')
            with open(save_file_path, 'w', encoding='UTF-8') as f:
                json.dump(output_dict, f, indent=0)
        else:
            save_file_path = os.path.join(out_dir, f'_{self.article_type}_labeled_sections.json')
            with open(save_file_path, 'w', encoding='UTF-8') as f:
                json.dump(output_dict, f, indent=0)
                print(f"Saved to: \n {save_file_path}")
                
        return save_file_path
   
    def _store_results(self, filename, labeled_sections):
        """
        Stores the labeled sections into the output dictionary and saves them to a JSON file.

        Parameters:
        - filename (str): The name of the file being processed.
        - labeled_sections (dict): The labeled sections of the text.
        
        Returns:
        - None
        """
        filename = os.path.splitext(os.path.basename(filename))[0]
        self.save_to_json({filename: labeled_sections}, filename=filename)
        self.output_dict[filename] = labeled_sections

    def _label_sections(self, text, question=None):
        """
        Labels sections of the text based on the article type.

        Parameters:
        - text (str): The text to be labeled.
        - question (str): The question for 'other' article type.

        Returns:
        - dict: Labeled sections of the text.
        """
        if self.article_type == 'research':
            labeled_sections, text = self.label_text(text)
        elif self.article_type == 'case':
            questions = self.get_questions()
            evaluator = CaseReportLabeler(api_key_path=self.api_key_path, text=text, questions=questions, section_headers=self.section_headers,question_token_estimate=500)
            labeled_sections = evaluator.evaluate_all_files()
        elif self.article_type == 'emr':
            questions = self.get_questions()
            # print(f'asking chatGPT with params is_azure={self.is_azure}, api_base={self.api_base}, api_version={self.api_version}')
            evaluator = CaseReportLabeler(api_key_path=self.api_key_path, text=text, questions=questions, section_headers=self.section_headers,question_token_estimate=500, is_azure=self.is_azure, deployment_id=self.deployment_id, api_base=self.api_base, api_version=self.api_version)
            labeled_sections = evaluator.evaluate_all_files()
        elif self.article_type == 'other':
            questions = self.get_questions(question)
            evaluator = CaseReportLabeler(api_key_path=self.api_key_path, text=text, questions=questions, section_headers=self.section_headers,question_token_estimate=500)
            labeled_sections = evaluator.evaluate_all_files()
        else:
            raise ValueError(f"Unknown article type {self.article_type}, choose 'case', 'research', 'emr', or 'other'")
        
        return labeled_sections

    def _json_file_exists(self, filename):
        """
        Checks if the JSON file for the given filename already exists.

        Parameters:
        - filename (str): The name of the file being processed.

        Returns:
        - bool: True if the JSON file exists, False otherwise.
        """
        root_dir = os.path.dirname(self.folder_path)
        out_dir = os.path.join(root_dir, 'json')
        json_filename = os.path.join(out_dir, f'{os.path.splitext(filename)[0]}_labeled_sections.json')
        return os.path.exists(json_filename)
    
    def _open_txt_file(self, filename):
        if self._json_file_exists(filename):
            print(f"Skipping {filename} as it is already processed.")
            return None
        try:
            with open(os.path.join(self.folder_path, filename), 'r', encoding='UTF-8') as f:
                text = f.read()
            return text
        except Exception as e:
            print(f"Failed to read file: {filename} ({e})")
            return None

    def skip_labeling(self):
        print("Skipping section labeling step.")
        self.output_dict={}
        filelist=os.listdir(self.folder_path)
        for filename in tqdm(filelist, desc='creating imitation labeling results', total=len(filelist)):
            if not filename.endswith('.txt'):
                continue
            with open(os.path.join(self.folder_path, filename), encoding='UTF-8') as file:
                processed=file.read().replace('|', ',') 
                self.output_dict[filename.split('.')[0]]={'emr':" ".join(processed.split())}

        # self.save_to_json(section_json)
        # os.mkdir(os.path.join(output_dir,'json'))
        # with open(json_file_path, 'w') as f:
        #     json.dump(section_json, f, indent=0)

    def process_files(self, question=None, skip_segmentation=False):
        """
        Processes all text files in the specified folder.
        
        TODO--this can be dramatically improved by saving a JSON file for each article, instead of a single large JSON. 
        To keep it compatible with susbequent code, could combine the JSONs after. 
        """
        if skip_segmentation:
            self.skip_labeling()
            self.save_to_json(self.output_dict)

        elif self._json_file_exists(self.article_type):

            print(f'_{self.article_type}_labeled_sections.json already exists, skipping processing. If you want to re-process, please delete this file first.')

        else:
            self.select_labels()
            file_list = os.listdir(self.folder_path)
            file_list = [f for f in file_list if f.endswith('.txt')]
            for filename in tqdm(file_list, desc='Segmenting text files',total=len(file_list)):
                text = self._open_txt_file(filename)
                if not text: continue
                labeled_sections = {}
                labeled_sections = self._label_sections(text, question)
                self._store_results(filename, labeled_sections)
            self.save_to_json(self.output_dict)

class FilterPapers:
    '''
    The FilterPapers class provides functionality to filter a dataset of articles based on inclusion/exclusion criteria.
    
    This class takes the paths to a CSV file and a JSON file as input. The CSV file should contain the articles that have passed the inclusion/exclusion criteria. The JSON file should contain the labeled sections of all the articles under consideration.
    
    The class offers methods to:
    1. Read the CSV and JSON files.
    2. Filter the JSON data based on the articles listed in the CSV.
    3. Save the filtered JSON data to a new JSON file.
    
    Attributes:
    - csv_path (str): Path to the CSV file containing articles that passed the inclusion/exclusion criteria.
    - json_path (str): Path to the JSON file containing labeled sections of all articles.
    - df (DataFrame): DataFrame containing articles that passed the inclusion/exclusion criteria.
    - data (dict): Dictionary containing the labeled sections of all articles.
    
    Example:
    csv_path = "/mnt/data/sample_articles.csv"
    json_path = "/mnt/data/labeled_sections.json"
    filter_papers = FilterPapers(csv_path=csv_path, json_path=json_path)
    filter_papers.run()
    '''
    def __init__(self, csv_path, json_path):
        """
        Initializes the FilterPapers class with paths to the CSV and JSON files.
        
        Parameters:
        - csv_path (str): Path to the CSV file containing the articles that passed the inclusion/exclusion criteria.
        - json_path (str): Path to the JSON file containing the labeled sections of all articles.
        """
        self.csv_path = csv_path
        self.json_path = json_path
        self.df = self.read_csv()
        self.data = self.read_json()

    def read_csv(self):
        """
        Reads the CSV file into a DataFrame. Assumes the first column of the CSV is the index.
        
        Returns:
        - DataFrame: DataFrame containing the articles that passed the inclusion/exclusion criteria.
        """
        return pd.read_csv(self.csv_path, index_col=0)

    def read_json(self):
        """
        Reads the JSON file into a dictionary.
        
        Returns:
        - dict: Dictionary containing the labeled sections of all articles.
        """
        with open(self.json_path, 'r', encoding='UTF-8') as file:
            return json.load(file)

    def filter_json(self):
        """
        Filters the JSON data based on the DataFrame. It selects only those articles that are present in the DataFrame's index.
        
        Returns:
        - dict: Dictionary containing the labeled sections of articles that passed the inclusion/exclusion criteria.
        """
        filtered_data = {key: value for key, value in self.data.items() if key in self.df.index}
        return filtered_data

    def save_to_json(self):
        """
        Saves the filtered JSON data to a new file in a directory called `filtered_articles`.
        
        Returns:
        - None
        """
        # Create a new directory in the same root folder
        out_dir = os.path.join(os.path.dirname(self.csv_path), "inclusion_exclusion_json")
        os.makedirs(out_dir, exist_ok=True)
        
        # Save the filtered dictionary to a JSON file
        with open(os.path.join(out_dir, 'filtered_labeled_sections.json'), 'w', encoding='UTF-8') as f:
            json.dump(self.filter_json(), f, indent=4)
        return os.path.join(out_dir, 'filtered_labeled_sections.json')

    def run(self):
        """
        A convenience method that calls `save_to_json()` to execute the filtering and saving in one step.
        
        Returns:
        - None
        """
        output_path = self.save_to_json()
        return output_path

class InclusionExclusionSummarizer:
    """
    Class to summarize inclusion/exclusion criteria based on the answers received from GPT-3.5.
    
    Attributes:
    - json_path (str): Path to the JSON file containing the answers.
    - data (dict): The data read from the JSON file.
    - df (DataFrame): Pandas DataFrame to store summarized results.
    """
    
    def __init__(self, json_path, questions, acceptable_strings=["1", "good", "excellent", "positive", " y " " y.", "yes", "1.0",  "correct", "is likely", "is possible", "is probable"]):
        """
        Initializes the InclusionExclusionSummarizer class.
        
        Parameters:
        - json_path (str): Path to the JSON file containing the answers.
        """
        self.acceptable_strings = acceptable_strings
        self.json_path = json_path
        self.questions = questions
        self.data = self.read_json()
        self.df = self.summarize_results()
    
    def read_json(self, json_file_path=None):
        """
        Reads JSON data from a file.
        
        Returns:
        - dict: The data read from the JSON file.
        """
        if json_file_path is None:
            json_file_path = self.json_path
        with open(json_file_path, 'r', encoding='UTF-8') as file:
            return json.load(file)
    
    def summarize_results(self):
        """
        Summarizes the results by converting answers to binary form.
        
        Returns:
        - DataFrame: Pandas DataFrame containing the summarized results.
        """
        summary_dict = {}
        for article, questions in self.data.items():
            summary_dict[article] = {}
            for question, chunks in questions.items():
                if question == 'metadata' or question.startswith('CHUNKS'):
                    continue
                # Get the polarity value for the question from the questions dictionary
                if question[:11] == 'EXPLANATION': 
                    summary_dict[article][question] = '|'.join([str(v) for v in chunks.values()])
                    continue
                # polarity = self.questions.get(question)
                # if polarity is None:
                #     raise ValueError(f"The question from the JSON: \n\n'{question}' \n\n was not found in the questions dictionary.")

                # Convert all chunk answers to lowercase and check for "yes" keywords
                binary_answers = [1 if any(s in answer.lower() for s in self.acceptable_strings) else 0 for answer in chunks.values()]
                
                # Set answer to 1 if any chunk is positive, otherwise 0
                summary_dict[article][question] = 1 if sum(binary_answers) > 0 else 0
        # Convert the summary dictionary to a DataFrame
        df = pd.DataFrame.from_dict(summary_dict, orient='index')
        
        # Set all values above 0 to 1
        # df[(type(df)!=str) and (df > 0)] = 1
        
        return df
    
    def drop_rows_with_zeros(self):
        """
        Drops any row in the DataFrame that contains a zero.
        
        Returns:
        - DataFrame: A new DataFrame with rows containing zeros removed.
        """
        return self.df[(self.df == 0).sum(axis=1) == 0]
    
    def save_to_csv(self, filename='inclusion_exclusion_results'):
        """
        Saves the DataFrame to a CSV file.
        
        Parameters:
        - dropped (bool): Indicates whether rows have been dropped from the DataFrame.
        
        Returns:
        - None
        """
        # Create a new directory in the same root folder
        out_dir = os.path.dirname(self.json_path)
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, filename + '.csv')
        self.df.index.name = 'MRN'
        self.df.to_csv(csv_path)
        return csv_path
            
    def run(self):
        """
        Executes all the summarization, saving and optional row-dropping steps in one method.
        
        Returns:
        - None
        """
        raw_path = self.save_to_csv()
        print(f"Your CSV files of filtered manuscripts have been saved to this directory: \n {os.path.dirname(raw_path)}")
        return self.df, raw_path


class CustomMapper():
    """
    Class to convert LLM responses into standardized answers, with several types of matching available.
    """
    def __init__(self, answer_format, string_mapping=None, 
                 summary_type='mapping', 
                 api_key_path=None, is_azure=False, deployment_id=None, api_base=None, api_version=None, max_workers=10, 
                ):
        """
        Initializes the CustomMapper class.
        """
        self.answer_format = answer_format
        self.summary_type=summary_type
        self.api_key_path = api_key_path
        self.is_azure = is_azure
        self.deployment_id = deployment_id
        self.api_base = api_base
        self.api_version = api_version
        self.max_workers = max_workers
        self.has_explanations = self.answer_format in ["binary_with_explanations", "binary_with_unknown_and_explanations","severity_with_unknown_and_explanations"]

        # --- NEW: allow severity mapping; else fall back to binary mapping if answers_binary=True ---
        if self.answer_format in ["binary_with_unknown_and_explanations", ]:

            if string_mapping is None:
                raise ValueError(f"Answer type {self.answer_format} cannot be evaluated without a string_mapping dict, but none was given")
            if not isinstance(string_mapping, dict) or len(string_mapping) == 0:
                raise ValueError("""Invalid severity mapping. "string_mapping" must be a non-empty dictionary""")
            
            # Normalize keys to ints and values to lowercase lists
            self.keyword_mapping = {
                int(k): [str(v).lower() for v in vals] for k, vals in string_mapping.items()
            }

        elif self.answer_format in ["binary","binary_with_explanations","binary_without_explanations"]:
            # if self.debug:
            #     print('Using binary mapping')
            self.keyword_mapping = {
                0: ["poor", "bad", "negative", "n", "no", "false", "absent"],
                1: ["good", "excellent", "positive", "y", "yes", "true", "present"]
            }
        elif self.answer_format=="severity_with_unknown_and_explanations":
            self.keyword_mapping=None
        else:
            raise ValueError('''"answer_format" is invalid. Allowed answer types are "binary_without_explanations", "binary_with_explanations", "binary_with_unknown_and_explanations","severity_with_unknown_and_explanations"''')

    @staticmethod
    def read_json(json_file_path):
        """Reads JSON data from a file and returns it as a dictionary."""
        try:
            with open(json_file_path, 'r', encoding='UTF-8') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Error: File {json_file_path} not found.")
            return {}
        except json.JSONDecodeError:
            print("Error: Could not decode the JSON file.")
            return {}

    @staticmethod
    def save_to_json(output_dict, save_file):
        """Saves dict to a JSON file."""
        with open(save_file, 'w', encoding='UTF-8') as f:
            json.dump(output_dict, f, indent=0)
        print(f"Saved to: {save_file}")

        return save_file

    def numerical_interpretation(self, answer):
        check_numeric_answer = re.sub(r'[^0-9\.\-\/]', '', answer)
        try:
            # if '/' in check_numeric_answer and any(ch.isdigit() for ch in check_numeric_answer):
            #     val = float(Fraction(check_numeric_answer))
            # else:
            val = float(check_numeric_answer)
            
            if self.answer_format=='severity_with_unknown_and_explanations':
                return val

            if float(val).is_integer() and float(val) in self.keyword_mapping.keys():
                return int(val)
            
        except Exception:
            pass

        return None

    def passthrough_without_mapping(self):
        # Raw text passthrough (research-style)
        summary_dict = {}

        for article, question_data in self.data.items(): # 'article' is subject id in EMR mode
            for question, responses in question_data.items(): # question is question text, responses is dict of chunk name to answer
                try:
                    combined_answers = list(responses.values())[0]
                    if not combined_answers:
                        summary_dict[article][question] = 'No Answers'
                    else:
                        summary_dict[article][question] = combined_answers
                except Exception as e:
                    summary_dict[article][question] = f'Error: {str(e)}'

        df = pd.DataFrame.from_dict(summary_dict, orient='index').fillna(np.nan)
        return df

    def exact_match(self, answer):
        """Checks for an exact match of keywords in the answer text."""
        if answer == 'Unidentified':
            return np.nan

        raw = str(answer).strip().lower()
        
        if self.answer_format=='severity_with_unknown_and_explanations': # for severity cases, prioritize numerical interpretation
            numerical=self.numerical_interpretation(raw)
            if numerical is not None:
                return numerical

        from string import punctuation
        raw_no_punctuation = raw.translate(str.maketrans(punctuation, ' '*len(punctuation)))

        found_keys_words = []
        for key, keywords in self.keyword_mapping.items():
            for keyword in keywords:

                if ' ' in keyword and keyword in raw_no_punctuation: # for multi-word keywords, check if the whole keyword is in the raw answer
                    found_keys_words.append((key, keyword))
                
                elif keyword in raw_no_punctuation.split(): # for 1-word keywords, check if any word matches exactly (to avoid partial matches)
                    found_keys_words.append((key, keyword))

        if len(set([k for k, w in found_keys_words]))==1: #If all found keywords correspond to the same key, return that key
            return found_keys_words[0][0]
        
        elif found_keys_words==[] and self.numerical_interpretation(answer) is None: # if we found no keywords, try numeric interpretation. If that fails, return None.
            return None

        # if we have multiple found keywords that correspond to different keys, 
        # we need to check if any of the keywords are substrings of each other (e.g. "no" in "no mention")
        good_keys_words = found_keys_words.copy() # copy of keys we found

        for small_key, small_word in found_keys_words: # loop through found keys and words
            for big_key, big_word in found_keys_words: # second loop
        
                if small_key != big_key and small_word in big_word: #if keys are different but words overlap, this is a problem
                    good_keys_words.remove((small_key, small_word)) # remove the shorter keyword (e.g. "no" in "no mention")
                    break # break inner loop since small_key has already been removed
        
        if len(set([k for k, w in good_keys_words]))==1: # Check for agreement again.
            return good_keys_words[0][0]
        else:
            return 99 # Indicates conflicting matches.

    def fuzzy_match(self, answer, threshold=60):
        """
        Fuzzy matches the answer with a list of keywords.

        Returns:
        - numeric level (int/float) or np.nan or None
        """
        cleaned = str(answer).lower()
        # numeric short-circuit
        try:
            val = float(cleaned)
            if val.is_integer():
                return int(val)
            return val
        except Exception:
            pass

        best_key = None
        highest_ratio = 0
        for key, keywords in self.keyword_mapping.items():
            for keyword in keywords:
                ratio = fuzz.ratio(cleaned, keyword.lower())
                if ratio > highest_ratio:
                    highest_ratio = ratio
                    best_key = key
        return best_key if highest_ratio >= threshold else None

    def keyword_or_fuzzy_match(self, answer):
        """
        Applies either exact matching or fuzzy matching based on the result of exact matching.

        Returns numeric level (binary or severity), np.nan, or None.
        """
        result = self.exact_match(answer)

        if result is None:
            result = self.fuzzy_match(answer)

        if result is None or np.isnan(result):
            result = 0

        return int(result)

    def interpret_left_right_severity(self, answer, divider='/'):
        """
        Interprets answers where left and right are (sometimes) given separately, separated by a slash/other divider.
        Common for BARS heel-shin, finger-nose, etc.
        """
        if divider in answer:
            left, right=[self.numerical_interpretation(score) for score in answer.split(divider)]
        else:
            left=self.interpret_severity(answer)
            right=left
        return left, right

    def interpret_severity(self, answer):
        mapped = self.numerical_interpretation(answer) 
        if mapped is None:
            mapped=99 
        return mapped

    def interpret_binary(self, raw_answer, article, question, chunk_name):

        chunk_answer=self.keyword_or_fuzzy_match(raw_answer)

        if chunk_answer not in self.keyword_mapping.keys() and chunk_answer != 99: # if the answer is not in our mapping and not the special "conflicting" code, this is a problem
            print(f"""Warning: unexpected answer "{chunk_answer}" from "{raw_answer}" for article "{article}", question "{question}", chunk "{chunk_name}". Setting to 99.""")    
            chunk_answer=99

        return chunk_answer

    def interpret_answer(self, raw_answer, article, q, chunk_name, is_left_right=False):
        if self.answer_format=='severity_with_unknown_and_explanations' and is_left_right:
            mapped=self.interpret_left_right_severity(raw_answer)

        elif self.answer_format=='severity_with_unknown_and_explanations':
            mapped=self.interpret_severity(raw_answer)

        else:
            mapped=self.interpret_binary(raw_answer, article, q, chunk_name)

        return mapped

    def map_json(self, input_json_path,output_json_path=None, left_right_questions=[]):
        """
        Takes the JSON output from OpenAIJsonEvaluator and maps the answers, 
        creating a new json with standardized values.
        """
        data=self.read_json(input_json_path)

        from copy import deepcopy
        mapped_dict = deepcopy(data)

        for article, question_data in tqdm(data.items(),"Interpreting answers: ",total=len(data)): 
            # 'article' is subject id in EMR mode

            for q, chunk_data in question_data.items():

                is_left_right=q in left_right_questions
                
                if q=='metadata' or 'explanation' in q.lower():
                    continue

                if is_left_right:
                    mapped_dict[article][q+'_left']={}
                    mapped_dict[article][q+'_right']={}

                for chunk_name, answer in chunk_data.items():

                    mapped_answer=self.interpret_answer(answer, article, q, chunk_name,
                                                        is_left_right=(q in left_right_questions))
                    if is_left_right:
                        mapped_dict[article][q+'_left'][chunk_name]=mapped_answer[0]
                        mapped_dict[article][q+'_right'][chunk_name]=mapped_answer[1]
                        mapped_dict[article][q][chunk_name]=max(mapped_answer) # set original question answer to whatever the max of the left and right scores was 

                    else:
                        mapped_dict[article][q][chunk_name]=mapped_answer

        if output_json_path is not None:
            self.save_to_json(mapped_dict, output_json_path)

        return mapped_dict

            
class CustomSummarizer(CustomMapper):
    """
    Class to create a custom summary of the results with user-defined keyword mapping and fuzzy matching.

    Attributes:
    - keyword_mapping (dict): Dictionary mapping each numeric level to a list of acceptable values.
        Examples:
          # Binary (kept for compatibility)
          {0: ["no","n","negative"], 1: ["yes","y","positive"]}

          # Severity scale
          {0: ["none","absent"], 1: ["mild"], 2: ["moderate"], 3: ["severe"]}
    """

    def __init__(self, json_path, answer_format, summary_type='llm', premapped=False, left_right_questions=[],
                 api_key_path=None, is_azure=False, deployment_id=None, api_base=None, api_version=None, max_workers=10, 
                 chunks_dir=None, string_mapping=None, debug=False,):
        """
        Initializes the CustomSummarizer class.

        Parameters:
        - json_path (str): Path to the JSON file containing the answers.
        - answers_binary (bool): If True, treat as binary (kept for compatibility). If False and
                                 string_mapping is provided, treat as severity scale.
        - string_mapping (dict|None): Optional numeric->list[str] mapping for severity levels.
                                        If provided, enables severity mode.
        """
        super().__init__(answer_format=answer_format, string_mapping=string_mapping)
        self.json_path = json_path
        self.summary_type = summary_type
        self.has_explanations = self.answer_format in ["binary_with_explanations", "binary_with_unknown_and_explanations","severity_with_explanations", "severity_with_unknown_and_explanations"]
        self.chunks_dir = chunks_dir
        self.api_key_path = api_key_path
        self.is_azure = is_azure
        self.deployment_id = deployment_id
        self.api_base = api_base
        self.api_version = api_version
        self.max_workers = max_workers
        self.debug = debug

        if answer_format=="severity_with_unknown_and_explanations":
            self.answer_categories=[-1,0,1,99]
        else:
            self.answer_categories=list(self.keyword_mapping.keys()) 

        if premapped:
            self.mapped_json_path = json_path
            self.data = self.read_json() 
        else:
            self.mapped_json_path=json_path.replace('.json','_mapped.json')
            self.data = self.map_json(json_path, 
                                      self.mapped_json_path,
                                      left_right_questions)


    @staticmethod
    def replace_long_question_names(df, questions_dict):
        
        new_cols=df.columns
        
        for short_q, long_q in questions_dict.items():
            new_cols=[col.replace(short_q, long_q) for col in new_cols]

        return new_cols

    @staticmethod
    def save_raw_results_to_csv(json_path, questions=None):
        """Saves the same information as the JSON file, but in CSV form.
        Each row is an individual note, with date and answers."""

        import pandas as pd

        output_dict=CustomSummarizer.read_json(json_path)
        subject_dfs=[]
        for subject, subject_data in output_dict.items():

            subdf=pd.DataFrame(subject_data)
            
            # subdf.reset_index(names='chunk', inplace=True)
            subdf.insert(0, 'MRN',subject)
            
            subdf.insert(1,'date', [metadata['date'] for metadata in subdf['metadata']])
            subdf.sort_values(by='date', inplace=True)
            subdf.drop(columns='metadata', inplace=True)

            subject_dfs.append(subdf)

        out=pd.concat(subject_dfs)
        out.sort_values(by=['MRN','date'], inplace=True)
        out.reset_index(names='chunk', inplace=True)
        
        mrn_col = out.pop("MRN")
        out.insert(0, 'MRN', mrn_col)

        if questions is not None:
            out.columns=CustomSummarizer.replace_long_question_names(out, questions)

        save_file=json_path.replace('.json','.csv')
        out.to_csv(save_file, index=False)

        return save_file

    def compile_answers(self, answers, article_name, question_name, has_failed_chunk=False):
                    
        valid_answers = [x for x in answers if (x is not None and not (isinstance(x, float) and np.isnan(x)))]
        if len(valid_answers) == 0:
            print(f'Warning: no valid responses for article "{article_name}" and question "{question_name}". Setting final answer to 0')
            return 0

        elif (np.nan in answers):
            if not has_failed_chunk:
                has_failed_chunk=True # only complains about failed chunks once for any subject
                print(f"Warning: Failed to interpret a chunk from '{article_name}'. The answers for that subject may be partially incorrect.")
            return np.nanmax(answers)

        else:
            return np.max(answers)
        
    def compile_text_data(self, data, data_name, positive_explanations_only=False):
        """Sorts, formats, and compiles explanations or chunks into a single string for each question."""
        output_dict={}
        for ans, text in data.items():
            if positive_explanations_only and ans == 0:
                continue
            elif text==[]:
                continue
            # sort by date. Works for both ranges and single dates,
            # but requires that the date is formatted as YYYY-MM-DD.
            text.sort(key = lambda x: x.split(':')[0].split()[0])
            # indents every new explanation for readability.
            output_dict[f"{ans}_{data_name}"] = "    "+'|\n    '.join(list(text))
        return output_dict

    def present_absent_category(self,value):
        """
        Determines whether to categorize a severity value as "present","absent", or "unknown".
        """
        if value==99:
            category=99
        if self.answer_format=="severity_with_unknown_and_explanations":
            category=int(value) if value <= 0 else 1 # if an answer is 0 for no or -1 for unknown, keep. if 1 or more, set to 'present'
        else:
            category=int(value) 
        return category


    def summarize_mapped_results(self, positive_explanations_only=False):
        """
        Summarizes the results based on keyword mapping/fuzzy matching.

        BEHAVIOR CHANGES:
        - Binary mode: if no positive evidence (sum == 0) ⇒ return NaN instead of 0 (your requested FP/FN handling).
        - Severity mode: aggregate by MAX severity over chunks (common choice). If no valid mapped chunks ⇒ NaN.
        """
        summary_dict = {}

        qs=list(self.data[next(iter(self.data))].keys()) # get question list from first article (assumes all have same questions)
        qs=[q for q in qs if (q != 'metadata' and 'EXPLANATION' not in q) and (not q.startswith('CHUNKS: '))] # filter out metadata and chunk content questions

        for article, question_data in tqdm(self.data.items(),"Summarizing answers: ",total=len(self.data)): # 'article' is subject id in EMR mode
        
            summary_dict[article] = {}
            has_failed_chunk=False
            chunk_metadatas=question_data['metadata']

            if self.chunks_dir is not None:
                chunks_dict = self.read_json(self.chunks_dir + '/' + article + '_chunks.json')

            for q in qs:
                answers=[]
                explanations={n:[] for n in self.answer_categories} # these two are now dicts, so that yes and no answers can be in separate cols
                saved_chunks={n:[] for n in self.answer_categories}
                
                for chunk_name, chunk_metadata in chunk_metadatas.items():

                    answer = question_data.get(q, {}).get(chunk_name, "")

                    answers.append(answer)

                    if self.has_explanations:
                        expl = question_data.get(f'EXPLANATION: {q}').get(chunk_name, "")
                        expl_category=self.present_absent_category(answer) # for severity, explanations for answers are grouped into "present" and "absent"
                        explanations[expl_category].append(f"{chunk_metadata['date_range']}, {answer}: {expl}")

                    if self.chunks_dir is not None:
                        expl_category=self.present_absent_category(answer)
                        saved_chunks[expl_category].append(f"{chunk_metadata['date_range']}: {chunks_dict[chunk_name]['text']}")

                summary_dict[article][q] = self.compile_answers(answers, article, q, has_failed_chunk)
                summary_dict[article].update(self.compile_text_data(explanations, f'explanations: {q}', positive_explanations_only))
                summary_dict[article].update(self.compile_text_data(saved_chunks, f'chunks: {q}', positive_explanations_only))

        df = pd.DataFrame.from_dict(summary_dict, orient='index').fillna(np.nan)

        return df

    def summarize_with_llm(self):
        summary_dict = {}
        import concurrent.futures

        def process_llm_summary(article, question, combined_answers):
            if question[:11] == 'EXPLANATION':
                return article, question, combined_answers
            else:
                question_formatted = (
                    f'Please summarize the following answers to the question: "{question}" '
                    f'Respond with 1 for yes or 0 for no, and do not include any explanations. '
                    f'If the answers are mixed or unclear, please respond with 0.'
                )
                summarizer = OpenAISummarizer(
                    api_key_path=self.api_key_path, text=combined_answers, question=question_formatted,
                    is_azure=self.is_azure, deployment_id=self.deployment_id,
                    api_base=self.api_base, api_version=self.api_version
                )
                val = summarizer.evaluate_text()
                try:
                    val = int(val)
                except Exception:
                    val = np.nan
                return article, question, (1 if val == 1 else np.nan)

        tasks = []
        for article, questions in self.data.items():
            summary_dict[article] = {}
            for question, chunks in questions.items():
                combined_answers = " ".join(str(answer) for answer in chunks.values())
                tasks.append((article, question, combined_answers))

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(process_llm_summary, *task): task for task in tasks}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc='Summarizing final responses with LLM'):
                article, question, result = future.result()
                summary_dict[article][question] = result

        return pd.DataFrame.from_dict(summary_dict, orient='index').fillna(np.nan)

    def summarize(self, positive_explanations_only=False):
        if self.summary_type == 'llm':
            df = self.summarize_with_llm()
        else:
            df = self.summarize_mapped_results(positive_explanations_only=positive_explanations_only)
        # else:
        #     df = self.passthrough_without_mapping()
        return df

    def run_custom(self, positive_explanations_only=False):
        """
        Executes all the summarization, saving, and optional row-dropping steps in one method.

        Returns:
        - DataFrame: Pandas DataFrame containing the summarized results.
        """
        self.df = self.summarize(positive_explanations_only=positive_explanations_only)
        raw_path = self.save_to_csv(filename=f'responses_raw')
        print(f"Your CSV files of filtered manuscripts have been saved to this directory: \n {os.path.dirname(raw_path)}")
        return self.df, raw_path

    def save_to_csv(self, filename='inclusion_exclusion_results'):
        """
        Saves the DataFrame to a CSV file.
        
        Parameters:
        - dropped (bool): Indicates whether rows have been dropped from the DataFrame.
        
        Returns:
        - None
        """
        # Create a new directory in the same root folder
        out_dir = os.path.dirname(self.json_path)
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, filename + '.csv')
        self.df.index.name = 'MRN'
        self.df.to_csv(csv_path)
        return csv_path