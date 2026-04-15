from tqdm import tqdm
from calvin_utils.gpt_sys_review.txt_utils import TextChunker
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
            with open(save_file_path, 'w') as f:
                json.dump(output_dict, f, indent=0)
        else:
            save_file_path = os.path.join(out_dir, f'_{self.article_type}_labeled_sections.json')
            with open(save_file_path, 'w') as f:
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
            with open(os.path.join(self.folder_path, filename), 'r') as f:
                text = f.read()
            return text
        except Exception as e:
            print(f"Failed to read file: {filename} ({e})")
            return None

    def skip_labeling(self):
        print("Skipping section labeling step.")
        self.output_dict={}
        for filename in os.listdir(self.folder_path):
            if not filename.endswith('.txt'):
                continue
            with open(os.path.join(self.folder_path, filename)) as file:
                processed=file.read().replace('|', ',') 
                self.output_dict[filename.split('.')[0]]={'emr':" ".join(processed.split())}

        # self.save_to_json(section_json)
        # os.mkdir(os.path.join(output_dir,'json'))
        # with open(json_file_path, 'w') as f:
        #     json.dump(section_json, f, indent=0)

    def process_files(self, question=None, label_files=True):
        """
        Processes all text files in the specified folder.
        
        TODO--this can be dramatically improved by saving a JSON file for each article, instead of a single large JSON. 
        To keep it compatible with susbequent code, could combine the JSONs after. 
        """
        if self._json_file_exists(self.article_type):

            print(f'_{self.article_type}_labeled_sections.json already exists, skipping processing. If you want to re-process, please delete this file first.')
        elif label_files:
            
            self.select_labels()
            file_list = os.listdir(self.folder_path)
            file_list = [f for f in file_list if f.endswith('.txt')]
            for filename in tqdm(file_list, desc='Segmenting text files'):
                text = self._open_txt_file(filename)
                if not text: continue
                labeled_sections = {}
                labeled_sections = self._label_sections(text, question)
                self._store_results(filename, labeled_sections)
            self.save_to_json(self.output_dict)
        else:
            self.skip_labeling()
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
        with open(self.json_path, 'r') as file:
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
        with open(os.path.join(out_dir, 'filtered_labeled_sections.json'), 'w') as f:
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
        with open(json_file_path, 'r') as file:
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
    
class CustomSummarizer(InclusionExclusionSummarizer):
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

    def __init__(self, json_path, answer_format, api_key_path=None, summary_type='llm',
                 chunks_dir=None, is_azure=False, deployment_id=None, api_base=None, api_version=None,
                 severity_mapping=None, max_workers=10, debug=False):
        """
        Initializes the CustomSummarizer class.

        Parameters:
        - json_path (str): Path to the JSON file containing the answers.
        - answers_binary (bool): If True, treat as binary (kept for compatibility). If False and
                                 severity_mapping is provided, treat as severity scale.
        - severity_mapping (dict|None): Optional numeric->list[str] mapping for severity levels.
                                        If provided, enables severity mode.
        """
        self.json_path = json_path
        self.api_key_path = api_key_path
        self.summary_type = summary_type
        self.answer_format = answer_format
        self.data = self.read_json()
        self.chunks_dir = chunks_dir
        self.is_azure = is_azure
        self.deployment_id = deployment_id
        self.api_base = api_base
        self.api_version = api_version
        self.max_workers = max_workers
        self.debug = debug

        # --- NEW: allow severity mapping; else fall back to binary mapping if answers_binary=True ---
        # self.severity_mode = False
        if self.answer_format in ["binary_with_unknown_and_explanations", "severity_with_explanations"]:

            self.severity_mode=True

            if severity_mapping is None:
                raise ValueError(f"Answer type {self.answer_format} cannot be evaluated without a severity_mapping dict, but none was given")
            if not isinstance(severity_mapping, dict) or len(severity_mapping) == 0:
                raise ValueError("""Invalid severity mapping. "severity_mapping" must be a non-empty dictionary""")
            
            # Normalize keys to ints and values to lowercase lists
            self.keyword_mapping = {
                int(k): [str(v).lower() for v in vals] for k, vals in severity_mapping.items()
            }

        elif self.answer_format in ["binary","binary_with_explanations","binary_without_explanations"]:
            self.severity_mode=False
            # if self.debug:
            #     print('Using binary mapping')
            self.keyword_mapping = {
                0: ["poor", "bad", "negative", "n", "no", "false", "absent", "No"],
                1: ["good", "excellent", "positive", "y", "yes", "true", "present", "Yes"]
            }
        else:
            raise ValueError('''"answer_format" is invalid. Allowed answer types are "binary_without_explanations", "binary_with_explanations", "binary_with_unknown_and_explanations","severity_with_explanations"''')


    def exact_match(self, answer):
        """Checks for an exact match of keywords in the answer text."""
        if answer == 'Unidentified':
            return np.nan

        raw = str(answer).strip().lower()

        # --- NEW: explicit binary mapping first ---
        # This catches pure yes/no outputs for binary-style questions.
        # if raw in ("Yes", "yes", "y", "true", "1"):
        #     return 1
        # if raw in ("No", "no", "n", "false", "0"):
        #     return 0
        # ------------------------------------------

        # Keep digits, dot, minus, and slash (so we can handle fractions like 1/4)
        cleaned_answer = re.sub(r'[^0-9\.\-\/]', '', raw)

        # Try numeric interpretation (for scales like 0/1/2/3, or "1/4", etc.)
        try:
            if '/' in cleaned_answer and any(ch.isdigit() for ch in cleaned_answer):
                val = float(Fraction(cleaned_answer))
            else:
                val = float(cleaned_answer)

            if float(val).is_integer():
                return int(val)
            return val
        except Exception:
            pass

        # Fall back to keyword mapping (severity words etc.)
        for key, keywords in self.keyword_mapping.items():
            for keyword in keywords:
                if keyword in raw.split():
                    return key
        return None

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
        exact_result = self.exact_match(answer)
        return exact_result if exact_result is not None else self.fuzzy_match(answer)
    

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

    def compile_answers(self, answers, article_name, question_name):
                    
        valid_answers = [x for x in answers if (x is not None and not (isinstance(x, float) and np.isnan(x)))]
        if len(valid_answers) == 0:
            print(f'Warning: no valid responses for article "{article_name}" and question "{question_name}". Setting final answer to 0')
            return 0

        elif (np.nan in answers):
            if not self.has_failed_chunk:
                self.has_failed_chunk=True # only complains about failed chunks once for any subject
                print(f"Warning: Failed to interpret a chunk from '{article_name}'. The answers for that subject may be partially incorrect.")
            return np.nanmax(answers)

        else:
            return np.max(answers)
        

    def summarize_results_with_mapping(self, positive_explanations_only=False):
        """
        Summarizes the results based on keyword mapping/fuzzy matching.

        BEHAVIOR CHANGES:
        - Binary mode: if no positive evidence (sum == 0) ⇒ return NaN instead of 0 (your requested FP/FN handling).
        - Severity mode: aggregate by MAX severity over chunks (common choice). If no valid mapped chunks ⇒ NaN.
        """
        summary_dict = {}

        qs=list(self.data[next(iter(self.data))].keys()) # get question list from first article (assumes all have same questions)
        qs=[q for q in qs if (q != 'metadata' and 'EXPLANATION' not in q) and (not q.startswith('CHUNKS: '))] # filter out metadata and chunk content questions
        # for article, questions in self.data.items(): # 'article' is subject id in EMR mode
        for article, question_data in self.data.items(): # 'article' is subject id in EMR mode
        
            summary_dict[article] = {}
            self.has_failed_chunk=False
            chunk_metadatas=question_data['metadata']

            if self.chunks_dir is not None:
                chunks_dict = self.read_json(self.chunks_dir + '/' + article + '_chunks.json')

            for q in qs:
                answers=[]
                explanations={}
                saved_chunks={}
                
                for chunk_name, chunk_metadata in chunk_metadatas.items():
                    chunk_answer = self.keyword_or_fuzzy_match(question_data.get(q).get(chunk_name, ""))

                    if np.isnan(chunk_answer):
                        chunk_answer=0
                    answers.append(int(chunk_answer))

                    if self.answer_format in ["binary_with_explanations","binary_with_unknown_and_explanations","severity_with_explanations"]:
                        expl = question_data.get(f'EXPLANATION: {q}').get(chunk_name, "")
                        
                        if int(chunk_answer) not in explanations.keys():
                            explanations[int(chunk_answer)] = []

                        explanations[int(chunk_answer)].append(f"{chunk_metadata['date_range']}: {expl}")

                    if self.chunks_dir is not None:
                        this_chunk = chunks_dict.get(chunk_name).get('text')

                        if int(chunk_answer) not in saved_chunks.keys():
                            saved_chunks[int(chunk_answer)] = []

                        saved_chunks[int(chunk_answer)].append(f"{chunk_metadata['date_range']}: {this_chunk}")

                summary_dict[article][q] = self.compile_answers(answers, article, q)

                for ans, expls in explanations.items():
                    if positive_explanations_only and ans == 0:
                        continue
                    summary_dict[article][f"{ans}_explanations: {q}"] = '   |   '.join(list(expls))
    
                for ans, q_chunks in saved_chunks.items():
                    if positive_explanations_only and ans == 0:
                        continue
                    summary_dict[article][f"{ans}_chunks: {q}"] = ' | '.join(list(q_chunks))

        df = pd.DataFrame.from_dict(summary_dict, orient='index').fillna(np.nan)

        return df


            # for question, responses in questions.items(): # question is question text, responses is dict of chunk name to answer
            #     if question == 'metadata' or question.startswith('CHUNKS'):
            #         continue
            #     # Keep explanations untouched
            #     if question[:11] == 'EXPLANATION' and positive_explanations_only:
            #         # Only keep explanations for “positive” responses
            #         # Note: mapped_answers is the answers for the previous question at this point, ie
            #         # the numerical answers which this answer is explaining. We can't determine if the answer
            #         # is positive from the explanations, so we use the numbers. Ugly but works for now.
                    
            #         retained_explanations= [
            #             f"{m}: {expl}" for expl, m in zip(responses.values(), mapped_answers)
            #             if (m is not None and m is not np.nan and m > 0)
            #         ]    
            #         summary_dict[article][question] = '|'.join(retained_explanations)

            #     elif question[:11] == 'EXPLANATION':
            #         summary_dict[article][question] = '|'.join(list(responses.values()))

            #     elif 'dates of the clinical notes' in question or 'dates' in question.lower() and ('Y/N' not in question and 'Yes/No' not in question):
            #         # Bypass binary keyword mapping for date extraction questions
            #         summary_dict[article][question] = '|'.join([str(v) for v in responses.values()])
            #         continue
                
            #     elif self.keyword_mapping:
            #         mapped_answers = [self.keyword_or_fuzzy_match(ans) for ans in responses.values()]
            #         # if any mapped is np.nan while others are valid, warn
            #         if (np.nan in mapped_answers) and not all(x is np.nan for x in mapped_answers):
            #             if has_failed_chunk:
            #                 continue
            #             has_failed_chunk=True # only complains about failed chunks once for any subject
            #             print(f"Warning: Failed to interpret a chunk from '{article}'. The answers for that subject may be partially incorrect.")

            #         valid_answers = [x for x in mapped_answers if (x is not None and not (isinstance(x, float) and np.isnan(x)))]

            #         if self.debug:
            #             print(valid_answers)
            #         if len(valid_answers) == 0:
            #             print(f'Warning: no valid responses for question "{question}". Setting final answer to 0')
            #             summary_dict[article][question] = 0
                        
            #         else:
            #             agg_value = np.max(valid_answers)
                        
            #             summary_dict[article][question] = agg_value
            #             if self.chunks_dir is not None:
            #                 # store chunks that contributed > 0 severity
            #                 retained_chunks = [chunks_dict[f'chunk_{i+1}'] for i, m in enumerate(mapped_answers)
            #                                 if (m is not None and not (isinstance(m, float) and np.isnan(m)) and m > 0)]
            #                 summary_dict[article]['CHUNKS: '+question] = '\n|\n'.join(retained_chunks)
                            

            #     elif self.keyword_mapping is None:
            #         # Raw text passthrough (research-style)
            #         try:
            #             combined_answers = list(responses.values())[0]
            #             if not combined_answers:
            #                 summary_dict[article][question] = 'No Answers'
            #             else:
            #                 summary_dict[article][question] = combined_answers
            #         except Exception as e:
            #             summary_dict[article][question] = f'Error: {str(e)}'
            #     else:
            #         raise ValueError("Unacceptable keyword mapping value.")

        # Build DataFrame; DO NOT collapse to 0/1 here (so NaN and severity survive)
        # df = pd.DataFrame.from_dict(summary_dict, orient='index').fillna(np.nan)
        # return df

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
        elif self.keyword_mapping:
            df = self.summarize_results_with_mapping(positive_explanations_only=positive_explanations_only)
        else:
            df = self.passthrough_without_mapping()
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