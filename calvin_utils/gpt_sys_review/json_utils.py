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

    def __init__(self, folder_path, article_type, api_key_path=None):
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
            evaluator = CaseReportLabeler(api_key_path=self.api_key_path, text=text, questions=questions, section_headers=self.section_headers,question_token_estimate=150)
            labeled_sections = evaluator.evaluate_all_files()
        elif self.article_type == 'emr':
            questions = self.get_questions()
            evaluator = CaseReportLabeler(api_key_path=self.api_key_path, text=text, questions=questions, section_headers=self.section_headers,question_token_estimate=150)
            labeled_sections = evaluator.evaluate_all_files()
        elif self.article_type == 'other':
            questions = self.get_questions(question)
            evaluator = CaseReportLabeler(api_key_path=self.api_key_path, text=text, questions=questions, section_headers=self.section_headers,question_token_estimate=150)
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

    def process_files(self, question=None):
        """
        Processes all text files in the specified folder.
        
        TODO--this can be dramatically improved by saving a JSON file for each article, instead of a single large JSON. 
        To keep it compatible with susbequent code, could combine the JSONs after. 
        """
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
    
    def __init__(self, json_path, questions, acceptable_strings=["1", "good", "excellent", "positive", " y " " y.", "yes", "correct", "is likely", "is possible", "is probable"]):
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

    
    def read_json(self):
        """
        Reads JSON data from a file.
        
        Returns:
        - dict: The data read from the JSON file.
        """
        with open(self.json_path, 'r') as file:
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
                # Get the polarity value for the question from the questions dictionary
                polarity = self.questions.get(question)
                if polarity is None:
                    raise ValueError(f"The question from the JSON: \n\n'{question}' \n\n was not found in the questions dictionary.")

                # Convert all chunk answers to lowercase and check for "yes" keywords
                binary_answers = [polarity if any(s in answer.lower() for s in self.acceptable_strings) else abs(1 - polarity) for answer in chunks.values()]
                
                # Sum up the binary answers for each question
                summary_dict[article][question] = sum(binary_answers)
        
        # Convert the summary dictionary to a DataFrame
        df = pd.DataFrame.from_dict(summary_dict, orient='index')
        
        # Set all values above 0 to 1
        df[df > 0] = 1
        
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
        self.df.to_csv(csv_path)
        return csv_path
            
    def run(self):
        """
        Executes all the summarization, saving and optional row-dropping steps in one method.
        
        Returns:
        - None
        """
        raw_path = self.save_to_csv()
        automated_path = self.save_to_csv(dropped=True)
        print(f"Your CSV files of filtered manuscripts have been saved to this directory: \n {os.path.dirname(raw_path)}")
        return self.df, raw_path, automated_path
    
class CustomSummarizer(InclusionExclusionSummarizer):
    """
    Class to create a custom summary of the results with user-defined keyword mapping and fuzzy matching.
    
    Attributes:
    - keyword_mapping (dict): Dictionary mapping each key to a list of acceptable values.
    """
    
    def __init__(self, json_path, answers_binary=False, api_key_path=None, summary_type='llm'):
        """
        Initializes the CustomSummarizer class.
        
        Parameters:
        - json_path (str): Path to the JSON file containing the answers.
        - keyword_mapping (dict): Dictionary mapping each key to a list of acceptable values.
        """
        self.json_path = json_path
        self.api_key_path = api_key_path
        self.summary_type = summary_type
        self.answers_binary = answers_binary
        self.data = self.read_json()
        if self.answers_binary:
            self.keyword_mapping = {
            0: ["poor", "bad", "negative", "n", "no"],
            1: ["good", "excellent", "positive", "y", "yes"]
            }
        else:
            self.keyword_mapping = None
    
    def exact_match(self, answer):
        """Checks for an exact match of keywords in the answer text."""
        cleaned_answer = re.sub(r'[^\w\s]', '', answer.lower())
        for key, keywords in self.keyword_mapping.items():
            for keyword in keywords:
                if keyword.lower() in cleaned_answer.split():
                    return key
        return None
    
    def fuzzy_match(self, answer, threshold=60):
        """
        Fuzzy matches the answer with a list of keywords.
        
        Parameters:
        - answer (str): The answer text to be matched.
        - threshold (int): The similarity ratio threshold for a valid match.
        
        Returns:
        - int or np.nan or None: Returns the key if a match is found, otherwise None.
        """
        best_match = None
        highest_ratio = 0
        for key, keywords in self.keyword_mapping.items():
            for keyword in keywords:
                ratio = fuzz.ratio(answer.lower(), keyword.lower())
                if ratio > highest_ratio:
                    highest_ratio = ratio
                    best_match = key
        if highest_ratio >= threshold:
            return best_match
        else:
            return None
    
    def keyword_or_fuzzy_match(self, answer):
        """
        Applies either exact matching or fuzzy matching based on the result of exact matching.
        
        Parameters:
        - answer (str): The answer text to be matched.
        
        Returns:
        - int or np.nan or None: Returns the key if a match is found, otherwise None.
        """
        exact_result = self.exact_match(answer)
        return exact_result if exact_result is not None else self.fuzzy_match(answer)
    
    def summarize_results_with_mapping(self):
        """
        Summarizes the results based on keyword mapping and fuzzy matching.
        
        Returns:
        - DataFrame: Pandas DataFrame containing the summarized results.
        """
        summary_dict = {}
        for article, questions in self.data.items():
            summary_dict[article] = {}
            for question, chunks in questions.items():
                #Extract binary data and process
                if self.keyword_mapping:
                    mapped_answers = [self.keyword_or_fuzzy_match(answer) for answer in chunks.values()]
                    if all(x is np.nan for x in mapped_answers) or all(x is None for x in mapped_answers):
                        summary_dict[article][question] = np.nan
                    else:
                        valid_answers = [x for x in mapped_answers if x is not np.nan and x is not None]
                        summary_dict[article][question] = np.sum(valid_answers) if valid_answers else 'Unidentified'
                #Extract raw data for research articles
                elif self.keyword_mapping is None:
                    try:
                        combined_answers = tuple(chunks.values())
                        if not combined_answers:
                            summary_dict[article][question] = 'No Answers'
                        else:
                            summary_dict[article][question] = combined_answers
                    except Exception as e:
                        summary_dict[article][question] = f'Error: {str(e)}'
                else:
                    raise ValueError("Unacceptable keyword mapping value.")
        df = pd.DataFrame.from_dict(summary_dict, orient='index').fillna(np.nan)
        if self.answers_binary:
            # Set all values above 0 to 1
            df[df > 0] = 1
        return df
    
    def summarize_with_llm(self):
        summary_dict = {}
        for article, questions in tqdm(self.data.items(), desc='Summarizing final responses with LLM'):
            summary_dict[article] = {}
            for question, chunks in questions.items():
                combined_answers = " ".join(str(answer) for answer in chunks.values())
                summarizer = OpenAISummarizer(api_key_path=self.api_key_path, text=combined_answers, question=question)
                summary_dict[article][question] = summarizer.evaluate_text()
        return pd.DataFrame.from_dict(summary_dict, orient='index').fillna(np.nan)
    
    def summarize(self):
        if self.summary_type == 'llm':
            df = self.summarize_with_llm()
        else:
            df = self.summarize_results_with_mapping()
        return df
    
    def run_custom(self):
        """
        Executes all the summarization, saving, and optional row-dropping steps in one method.
        
        Returns:
        - DataFrame: Pandas DataFrame containing the summarized results.
        """
        self.df = self.summarize()
        raw_path = self.save_to_csv(filename=f'responses_raw')
        automated_path = self.save_to_csv(filename=f'responses_claned_with_{self.summary_type}')
        print(f"Your CSV files of filtered manuscripts have been saved to this directory: \n {os.path.dirname(raw_path)}")
        return self.df, raw_path, automated_path