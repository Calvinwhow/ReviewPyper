from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from tqdm import tqdm
from calvin_utils.gpt_sys_review.txt_utils import TextChunker
from calvin_utils.gpt_sys_review.gpt_utils import CaseReportLabeler
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
    - article_type (str): The type of article (e.g., 'research', 'case').
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
        - article_type (str): The type of article (e.g., 'research' or 'case').
        - folder_path (str): The path to the folder containing text files.
        - article_type (str): The type of article (e.g., 'research', 'case').
        """
        self.api_key_path = api_key_path
        self.folder_path = folder_path
        self.article_type = article_type
        self.chunker = None

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
        else:
            raise ValueError(f"Unknown article type {self.article_type}, choose case or research.")
            
    def get_questions(self):
        if self.article_type == "research":
            return None
        elif self.article_type == "case":
            questions = {'Priotizing implicit and explicit information, do you think this contain a neurological case report? For example, if the text refers to "the patient", referse to an individual, or part of a clinical investigation, this is a strong yes. (Y/N)': 'case_report'}
            return questions
        else:
            return None

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
        
    def process_files(self):
        """
        Processes all text files in the specified folder.
        """
        output_dict = {}

        self.select_labels()

        for filename in tqdm(os.listdir(self.folder_path)):
            if filename.endswith('.txt'):
                with open(os.path.join(self.folder_path, filename), 'r') as f:
                    text = f.read()

                # Initialize labeled_sections
                labeled_sections = {}

                # Use keyword matching for stereotypical articles
                if self.article_type == 'research':
                    labeled_sections, text = self.label_text(text)
                elif self.article_type == 'case':
                    questions = self.get_questions()
                    evaluator = CaseReportLabeler(api_key_path=self.api_key_path, text=text, questions=questions, section_headers=self.section_headers)
                    labeled_sections = evaluator.evaluate_all_files()
                else:
                    raise ValueError(f"Unknown article type {self.article_type}, choose case or research")

                output_dict[filename] = labeled_sections

        self.save_to_json(output_dict)

    def save_to_json(self, output_dict):
        """
        Saves the labeled sections to a JSON file.

        Parameters:
        - output_dict (dict): Dictionary containing the labeled sections.

        Returns:
        - None
        """
        # Create a new directory in the same root folders
        out_dir = os.path.join(self.folder_path, '..', 'labeled_text')
        os.makedirs(out_dir, exist_ok=True)
        save_file_path = os.path.join(out_dir, f'{self.article_type}_labeled_sections.json')
        with open(save_file_path, 'w') as f:
            json.dump(output_dict, f, indent=0)
        print(f"Saved to: \n {save_file_path}")
        return save_file_path

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
    
    def __init__(self, json_path, questions, acceptable_strings=["good", "excellent", "positive", " y " " y.", "yes", "correct", "is likely", "is possible", "is probable"]):
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
    
    def save_to_csv(self, dropped=False, filename='inclusion_exclusion_results'):
        """
        Saves the DataFrame to a CSV file.
        
        Parameters:
        - dropped (bool): Indicates whether rows have been dropped from the DataFrame.
        
        Returns:
        - None
        """
        # Create a new directory in the same root folder
        out_dir = os.path.join(os.path.dirname(self.json_path), filename)
        os.makedirs(out_dir, exist_ok=True)
        
        # Determine the name of the CSV file based on whether rows have been dropped
        file_name = "processed_results.csv" if dropped else "raw_results.csv"
        
        # Save the DataFrame to a CSV file
        csv_path = os.path.join(out_dir, file_name)
        if dropped:
            self.drop_rows_with_zeros().to_csv(csv_path)
        else:
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
    
    def __init__(self, json_path, answers_binary=False):
        """
        Initializes the CustomSummarizer class.
        
        Parameters:
        - json_path (str): Path to the JSON file containing the answers.
        - keyword_mapping (dict): Dictionary mapping each key to a list of acceptable values.
        """
        self.json_path = json_path
        self.data = self.read_json()
        self.answers_binary = answers_binary
        if self.answers_binary:
            self.keyword_mapping = {
            0: ["poor", "bad", "negative", "n", "no"],
            1: ["good", "excellent", "positive", "y", "yes"]
            }
        else:
            self.keyword_mapping = None
    
    def exact_match(self, answer):
        """
        Checks for an exact match of keywords in the answer text.
        
        Parameters:
        - answer (str): The answer text to be matched.
        
        Returns:
        - int or np.nan or None: Returns the key if a match is found, otherwise None.
        """
        import re
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
        from fuzzywuzzy import fuzz
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
    
    def run_custom(self):
        """
        Executes all the summarization, saving, and optional row-dropping steps in one method.
        
        Returns:
        - DataFrame: Pandas DataFrame containing the summarized results.
        """
        self.df = self.summarize_results_with_mapping()
        raw_path = self.save_to_csv(filename='data_extraction')
        automated_path = self.save_to_csv(dropped=True, filename='data_extraction')
        print(f"Your CSV files of filtered manuscripts have been saved to this directory: \n {os.path.dirname(raw_path)}")
        return self.df, raw_path, automated_path

class LabelWithLDA:
    """
    This class is deprecated and not currently supported.
    Uses Latent Dirichlet Allocation for NLP text labelling.
    """
    def train_lda(self, text_chunks):
        """
        Trains the LDA model based on the text chunks.

        Parameters:
        - text_chunks (list): List of text chunks.

        Returns:
        - object: Trained LDA model.
        """
        self.vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
        data_vectorized = self.vectorizer.fit_transform(text_chunks)
        lda_model = LatentDirichletAllocation(n_components=len(self.topic_labels), max_iter=10, learning_method='online')
        lda_Z = lda_model.fit_transform(data_vectorized)
        self.lda_model = lda_model
        return lda_model      
    
    def train_lda_on_all_files(self):
        """
        Trains the LDA model on text from all files in the specified folder.

        Returns:
        - object: Trained LDA model.
        """
        all_text_chunks = []

        for filename in tqdm(os.listdir(self.folder_path)):
            if filename.endswith('.txt'):
                with open(os.path.join(self.folder_path, filename), 'r') as f:
                    text = f.read()

                self.chunker = TextChunker(text, np.round(4096*0.75)) # Set to 75% max token limit
                self.chunker.chunk_text()
                chunks = self.chunker.get_chunks()

                all_text_chunks.extend(chunks)

        # Train LDA model on all text chunks
        return self.train_lda(all_text_chunks)  
        
    def dominant_topic(self, text_chunk):
        """
        Finds the dominant topic for a given text chunk.

        Parameters:
        - text_chunk (str): The text chunk to be labeled.

        Returns:
        - int: Index of the dominant topic.
        """
        text_vectorized = self.vectorizer.transform([text_chunk])
        topic_probability_scores = self.lda_model.transform(text_vectorized)
        dominant_topic_index = topic_probability_scores.argmax()
        return dominant_topic_index
    
    def label_with_lda(self, text_chunk):
        # Get the list of section labels based on the article type
        return self.topic_labels[self.dominant_topic(text_chunk)]
    
    def extract_topic_significant_words(self, dominant_topic_index):
        """
        Extracts significant words for a specific topic in the LDA model.

        Parameters:
        - dominant_topic_index (int): The index of the dominant topic.

        Returns:
        - list: Significant words for the dominant topic.
        """
        # Initialize an empty list to store the significant words for the dominant topic
        significant_words = []

        # Get the topic-word distribution for the dominant topic
        topic_word_distribution = self.lda_model.components_[dominant_topic_index]

        # Get the indices of the top N significant words
        N = 10  # Adjust as needed
        top_word_indices = topic_word_distribution.argsort()[-N:][::-1]

        # Get the actual words from the vectorizer
        feature_names = self.vectorizer.get_feature_names_out()
        significant_words = [feature_names[i] for i in top_word_indices]

        return significant_words