import openai
import json
import os 
import time
import sys
import numpy as np
from tqdm import tqdm
from calvin_utils.gpt_sys_review.txt_utils import TextChunker
    
class QuestionTemplate:
    """
    Class to manage and display templates for various types of questions.
    
    # Example usage
    question_template = QuestionTemplate()
    question_template.inclusion_exclusion_questions()
    question_template.data_extraction_questions()
    question_template.print_question_template("inclusion")
    question_template.print_question_template("exclusion")
    question_template.print_question_template("custom")
    """
    def __init__(self):
        self.get_question_templates(self)
    
    def inclusion_exclusion_questions(self):
        """Method to print inclusion and exclusion questions."""

        print("Here are example inclusion questions:")
        print(json.dumps(self.inclusion_questions, indent=4))
        print("Here are example exclusion questions:")
        print(json.dumps(self.exclusion_questions, indent=4))
        print("Here is a question template")
        print(json.dumps(self.template_questions, indent=4))
    
    def data_extraction_questions(self):
        """Method to print data extraction questions."""
        
        print("Here are example data extraction questions:")
        print(json.dumps(self.strict_extraction_questions, indent=4))
        print("Here are some more generalizable data extraction questions:")
        print(json.dumps(self.lenient_extraction_questions, indent=4))
        print("Here is a question template")
        print(json.dumps(self.template_questions, indent=4))
    
    def get_question_templates(self, question_type):
        """
        Prints out a template for questions based on the specified question type.
        
        Parameters:
        - question_type (str): Type of questions to print ('inclusion', 'exclusion', 'evaluation').
        
        Returns:
        None
        """
        self.inclusion_questions = {'Amnesia case report? (Y/N)': 'case_report',
                            'Published in English? (Y/N)': 'is_english'}
        self.exclusion_questions = {'Transient amnesia, reversible amnesia symptom, severe confabulation or drug use, toxicity, epilepsy-related confusion, psychological or psychiatric-related amnesia (functional amnesia)': 'other_cause',
                            'Did not examine/report both retrograde and anterograde memory domains': 'not_both_domains',
                            'Without descriptive/qualitative/quantitative data on amnesia severity/memory tests/questions/scenarios/details': 'not_enough_information',
                            'Had global cognitive impairment disproportionate to memory loss': 'disproportionate_impairment',
                            'Without measurable lesion-related brain MR/CT scans': 'no_scan',
                            'Had focal or widespread brain atrophy': 'neurodegenerative',
                            'Atypical cases with selective (e.g., semantic) memory loss or material/topographic-specific memory loss': 'atypical_case'
                        }  
        self.strict_extraction_questions = {'Does the patient(s) represent(s) the whole experience of the investigator (center) or is the selection method unclear to the extent that other patients with similar presentation may not have been reported? (Good/Bad/Unclear)': 'representative_case_quality',
                            'Was patient’s causal exposure clearly described? (Good/Bad/Unclear)': 'causality_quality',
                            'Were diagnostic tests or assessment methods and the results clearly described (amnesia tests)? (Good/Bad/Unclear)': 'phenotyping_quality',
                            'Were other alternative causes that may explain the observation (amnesia) ruled out? (Good/Bad/Unclear)': 'workup_quality',
                            'Were patient’s demographics, medical history, comobidities clearly described? (Good/Bad/Unclear)': 'clinical_covariates_quality',
                            'Were patient’s symptoms, interventions, and clinical outcomes clearly presented as a timeline? (Good/Bad/Unclear)': 'history_quality',
                            'Was the lesion image taken around the time of observation (amnesia) assessment? (Good/Bad/Unclear)': 'temporal_causality_quality',
                            'Is the case(s) described with sufficient details to allow other investigators to replicate the research or to allow practitioners make inferences related to their own practice? (Good/Bad/Unclear)': 'history_quality_2'
            }
        self.lenient_extraction_questions = {'How affected is this study by selection bias? Little (Good). Very (Bad). Unsure (Unclear).': 'selection_bias',
                            'How well was the timeline of lesion/symptom onset described? (Good/Bad/Unclear)': 'exposure',
                            'How affected is this case by attribution error? Little (Good). Very (Bad). Unsure (Unclear).': 'outcome',
                            'How reasonable is the attribution of the lesion/diagnosis to the symptom? (Good/Bad/Unclear)': 'attribution_error',
                            'How well was the patients baseline pre-lesion described? (Good/Bad/Unclear)': 'pre_event_assessment',
                            'How well was the patients outcome post-lesion described? (Good/Bad/Unclear)': 'post_event_assessment',
                            'Do you think the neuroimaging was taken within temporal proximity to the lesion? Days-weeks (Good). Months-years (Bad). Unsure (Unclear).': 'lesion_related-images',
                            'Do you think another practitioner would come to the same conclusion (diagnosis/attribution) that this group did? (Good/Bad/Unclear)': 'replicability',
                            'How is the quality of this case overall? (Good/Bad/Unclear)': 'overall_appraisal'
            }        
        self.template_questions = {' ? (metric/metric/metric)': 'question_label',
                            ' ? (metric/metric/metric)': 'question_label',
                            ' ? (metric/metric/metric)': 'question_label',
                            ' ? (metric/metric/metric)': 'question_label',
                            ' ? (metric/metric/metric)': 'question_label',
                            ' ? (metric/metric/metric)': 'question_label',
                            ' ? (metric/metric/metric)': 'question_label',
                            ' ? (metric/metric/metric)': 'question_label',
            }

        
class OpenAIEvaluator:
    """
    A class to evaluate text chunks using the OpenAI API based on the type of article.

    Attributes:
    - api_key (str): OpenAI API key.
    - article_type (str): The type of article (e.g., 'research', 'case').
    - questions (dict): Dictionary mapping article types to evaluation questions.

    Methods:
    - __init__: Initializes the OpenAIEvaluator class with the API key path and article type.
    - read_api_key: Reads the OpenAI API key from a file.
    - evaluate_with_openai: Evaluates a text chunk based on the question corresponding to the article type.
    """


    def __init__(self, api_key_path):
        """
        Initializes the OpenAIEvaluator class.

        Parameters:
        - api_key_path (str): Path to the file containing the OpenAI API key.
        - article_type (str): The type of article (e.g., 'research', 'case').
        """
        self.api_key = self.read_api_key(api_key_path)
        openai.api_key = self.api_key

    def read_api_key(self, file_path):
        """
        Reads the OpenAI API key from a file.

        Parameters:
        - file_path (str): Path to the file containing the OpenAI API key.

        Returns:
        - str: OpenAI API key.
        """
        with open(file_path, 'r') as file:
            return file.readline().strip()

    def evaluate_with_openai(self, chunk, questions):
        """
        Evaluates a chunk based on multiple posed questions using OpenAI API.

        Parameters:
        - chunk (str): The text chunk to be evaluated.
        - questions (list): A list of questions for evaluation.

        Returns:
        - dict: A dictionary where keys are questions and values are binary decisions (0 or 1).
        """
        question_list = list(questions.keys())
        question_prompt = "\n".join([f"{q}" for q in question_list])
        prompt = f"Text Chunk: {chunk}\n{question_prompt}"

        try:
            response = openai.Completion.create(
                engine="gpt-3.5-turbo-16k-0613",
                prompt=prompt,
                max_tokens=10  # Adjust as needed
            )
            decision_text = response.choices[0].text.strip()
            decisions = decision_text.split("\n")
            
            if len(decisions) != len(questions):
                print("Warning: The number of decisions does not match the number of questions.")
                valid_decisions = [line.strip() for line in decisions if line.strip()]
                if len(valid_decisions) != len(questions):
                    decisions = valid_decisions
                    print("Solved warning.")
                else:
                    print('Decisions here, returning None: ', decisions)
                    return None

            return {q: 1 if "Y" in d else 0 for q, d in zip(questions, decisions)}
            
        except Exception as e:
            print(f"An error occurred: {e}")
            return "Unidentified"

class OpenAIChatEvaluator(OpenAIEvaluator):
    """
    Class to evaluate text chunks using OpenAI's chat models.
    
    Attributes:
    - token_limit (int): The maximum number of tokens allowed in each OpenAI API call.
    - question_token (int): The number of tokens reserved for the question.
    - answer_token (int): The number of tokens reserved for the answer.
    - json_data (dict): The data read from the JSON file.
    - keys_to_consider (list): List of keys to consider from the JSON file.
    - article_type (str): The type of article (e.g., 'research', 'case').
    - questions (dict): Dictionary mapping article types to evaluation questions.
    """
    
    def __init__(self, api_key_path, json_file_path, keys_to_consider, question_type, question, model_choice="gpt3_small", debug=False, test_mode=True):
        """
        Initializes the OpenAIChatEvaluator class.
        
        Parameters:
        - api_key_path (str): Path to the file containing the OpenAI API key.
        - json_file_path (str): Path to the JSON file containing the text data.
        - keys_to_consider (list): List of keys to consider from the JSON file.
        - article_type (str): The type of article (e.g., 'research', 'case').
        - token_limit (int): The maximum number of tokens allowed in each OpenAI API call. Default is 16000.
        - question_token (int): The number of tokens reserved for the question. Default is 500.
        - answer_token (int): The number of tokens reserved for the answer. Default is 500.
        - test_mode (bool): Will only pass the first article to GPT. Used to iteratively refine the passed questions.
        """
        super().__init__(api_key_path)
        self.questions = question
        self.json_path = json_file_path
        self.json_data = self.read_json(json_file_path)
        self.keys_to_consider = keys_to_consider
        self.question_type = question_type
        self.all_answers = {}
        self.debug = debug
        self.get_model_data(model_choice)
        self.get_question_settings(question_type)
        
        self.test_mode = test_mode
        if self.test_mode and self.json_data:
            first_key = next(iter(self.json_data.keys()))
            self.json_data = {first_key: self.json_data[first_key]}
            print(f'Will evaluate only {len(self.json_data)} articles for testing.')
        self.extract_relevant_text()
        
    def get_question_settings(self, question_type):
        """
        Sets the manner in which directives and questions are posed to the model.
        """
        if self.question_type=="research":
            self.directive = "You are a research assistant. Your task is to carefully evaluate the following research report. Use both explicit information and reasonable inferences to answer the questions. Be as concise as possible."
            self.chunk_flag = "[RESEARCH REPORT]"
            self.dir = "research_extractions"
        elif self.question_type=="case":
            self.directive = "You are a medical assistant. Your task is to carefully evaluate the following case report. Use both explicit information and reasonable inferences to answer the questions. Be as concise as possible."
            self.chunk_flag = "[CASE REPORT]"
            self.dir = "case_extractions"
        elif self.question_type=="labelling":
            self.directive = "You are a text labelling assistant. Your task is to carefully evaluate the following case report. Use both explicit information and reasonable inferences to answer the questions. Be as concise as possible."
            self.chunk_flag = "[SEGMENT]"
            self.dir = "labelling_extractions"
            self.token_limit = 500
        else:
            raise ValueError(f"Model choice {question_type} not supported, please choose gpt4, gpt3_large, or gpt3_small.")
        
    def get_model_data(self, model_choice):
        """
        Sets values for the OpenAI model to use.
        """
        self.temperature = 1.0
        self.response_tokens = 50
        self.question_token = 500
        # Assign Model-Specific Values
        if model_choice=="gpt4":
            self.model = "gpt-4"
            self.token_limit = 8192 - 2*(self.question_token)
            self.cost = 0.03/1000
        elif model_choice=="gpt3_large":
            self.model = "gpt-3.5-turbo-16k"
            self.token_limit = 16385 - 2*(self.question_token)
            self.cost = 0.003/1000
        elif model_choice=="gpt3_small":
            self.model = "gpt-3.5-turbo"
            self.token_limit = 4097 - np.round(1.2*(self.question_token))
            self.cost = 0.0015/1000
        else:
            raise ValueError(f"Model choice {model_choice} not supported, please choose gpt4, gpt3_large, or gpt3_small.")

    def read_json(self, json_file_path):
        """
        Reads JSON data from a file.
        
        Parameters:
        - json_file_path (str): Path to the JSON file containing the text data.
        
        Returns:
        - dict: The data read from the JSON file.
        """
        try:
            with open(json_file_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Error: File {json_file_path} not found.")
            return {}
        except json.JSONDecodeError:
            print("Error: Could not decode the JSON file.")
            return {}
    
    def extract_relevant_text(self):
        """
        Extracts and stores relevant text sections based on keys_to_consider.
        """
        self.relevant_text_by_file = {}
        for file_name, sections in self.json_data.items():
            selected_text = ""
            for key, value in sections.items():
                if key in self.keys_to_consider:
                    selected_text += value
            self.relevant_text_by_file[file_name] = selected_text
            
    def call_chunker(self, selected_text):
        """
        Uses TextChunker defined in text_utils.py to extract text in chunks
        """
        text_chunker = TextChunker(selected_text, self.token_limit)
        text_chunker.chunk_text()
        chunks = text_chunker.get_chunks()
        if self.debug:
                print('Text associated with file:', selected_text)
                print(f'Allowing {self.token_limit} tokens per submission')
                print('Number of chunks:', len(chunks))
        return chunks
    
    def generate_submission(self, chunk, q):
        """
        Prepares the submission to the OpenAI Model
        """
        conversation = [{"role": "system", "content": f"{self.directive}"},
                        {"role": "user", "content": f"{self.chunk_flag}: {chunk}"}]
        if (self.model == "gpt-4") or (self.model == "gpt-3.5-turbo-16k") or (self.model == "gpt-3.5-turbo"):
            conversation.append({"role": "user", "content": f'Based on the {self.chunk_flag} provided, {q}'})
        else:
            raise ValueError(f"{self.model} not yet supported.")
        return conversation
    
    def get_response_from_openai(self, conversation):
        """
        Sends a conversation to OpenAI and retrieves the assistant's last answer.

        Parameters:
        - conversation (list): The conversation history, including the system message, user question, and assistant's response if any.

        Returns:
        - str: The assistant's last answer retrieved from OpenAI's API.
        """
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=conversation,
            temperature=self.temperature,
            max_tokens=self.response_tokens
        )
        if self.debug:
            print('Handling OpenAI Response')
        return response['choices'][-1]['message']['content'], response["usage"]["total_tokens"]
        
    def handle_response_exception(self, e, q_index, retry_count):
        """
        Handles exceptions during API calls to OpenAI.

        Parameters:
        - e (Exception): The exception object.
        - q_index (int): The index of the current question being processed.
        - retry_count (int): The current count of retry attempts.

        Returns:
        - int: The updated retry_count.
        - int: The sleep time in seconds before the next retry.
        """
        if type(e).__name__ == 'RateLimitError':
            print(f"Rate limit error: {e}. Retrying Question No. {q_index} Attempt:({retry_count+1})")
            return retry_count + 1, 30
        else:
            print(f"An error occurred: {e}. Retrying Question No. {q_index} Attempt:({retry_count+1})")
            return retry_count + 1, 1

    def evaluate_all_files(self):
        try:
            for file_name, selected_text in tqdm(self.relevant_text_by_file.items()):
                # Chunk text by token limits
                chunks = self.call_chunker(selected_text)
                if self.debug:
                    print('On file:', file_name)
                
                # Initialize a dictionary to store chunk-level answers for each question
                self.all_answers[file_name] = {}
                for question in self.questions.keys():
                    self.all_answers[file_name][question] = {}

                # Send a query for each chunk
                for chunk_index, chunk in enumerate(chunks):
                    if self.debug:
                        print(f'On chunk: {chunk_index}/{len(chunks)}')
                        print(f'Text in chunks: {chunk}')
                    # Initialize a conversation with OpenAI for this chunk
                    for q_index, q in enumerate(self.questions.keys()):
                        
                        # Generate the conversation to submit
                        conversation = self.generate_submission(chunk, q)
                        
                        # Use a while loop to allow 3 submission attempts
                        retry_count = 0
                        while retry_count < 4:
                            try:
                                if self.debug:
                                    print('Submitting conversation to OpenAI')
                                answer, tokens_used = self.get_response_from_openai(conversation)
                                if self.debug:
                                    print(f"Q: {q} \n A: {answer}")
                                    
                                # Store the answer for this question and this chunk
                                self.all_answers[file_name][q][f"chunk_{chunk_index+1}"] = answer
                                
                                time.sleep(0.1)
                                break  # Exit the loop if successful
                            
                            #Handle Exceptions
                            except Exception as e:
                                if retry_count == 3:
                                    print(f"Exceeded 3 attempts due to error: \n {e}")
                                    self.all_answers[file_name][q][f"chunk_{chunk_index+1}"] = "Unidentified"
                                retry_count, sleep_time = self.handle_response_exception(e, q_index, retry_count)
                                time.sleep(sleep_time)

                if self.test_mode:
                    print(f'{tokens_used} tokens counted by the OpenAI API. Estimated cost per article: {tokens_used*self.cost*len(self.questions.items())*len(chunks)}')
            return self.all_answers
        except KeyboardInterrupt:
            print("KeyboardInterrupt detected. Saving results and closing.")
            self.save_to_json(self.all_answers)
            sys.exit(0)

    def save_to_json(self, output_dict):
        """
        Saves the labeled sections to a JSON file.

        Parameters:
        - output_dict (dict): Dictionary containing the labeled sections.

        Returns:
        - None
        """
        # Create a new directory in the same root folder
        out_dir = os.path.join(os.path.dirname(self.json_path), '..', f"{self.dir}")
        os.makedirs(out_dir, exist_ok=True)
        
        # Save the dictionary to a JSON file
        save_file = os.path.join(out_dir, f'{self.question_type}_evaluations.json')
        with open(save_file, 'w') as f:
            json.dump(output_dict, f, indent=0)
        print(f"Saved to: {save_file}")
        return save_file
    
class CaseReportLabeler(OpenAIChatEvaluator):
    def __init__(self, api_key_path, text, questions, section_headers):
        self.text = text
        self.section_headers = section_headers
        super().__init__(api_key_path, 
                         json_file_path=None, 
                         keys_to_consider=None, 
                         question_type="labelling", 
                         question=questions, 
                         model_choice="gpt3_small",
                         debug=False, 
                         test_mode=False)


    def extract_relevant_text(self):
        self.relevant_text_by_file = {"file_1": self.text}

    def read_json(self, json_file_path=None):
        """
        Overrides read_json method in parent class.
        """
        return {}
    
    def evaluate_all_files(self):
        """
        Evaluates all the text files to categorize text chunks based on the answers to questions.

        Returns:
        - results_dict (dict): Dictionary containing text chunks categorized under keys from section_headers.
        """
        results_dict = {'case_report': [], 'other': []}
        acceptable_case_answers = self.section_headers.get('Case_Report', [])
        
        for file_name, selected_text in tqdm(self.relevant_text_by_file.items()):
            # Chunk text by token limits
            chunks = self.call_chunker(selected_text)
            
            # Send a query for each chunk
            for chunk_index, chunk in enumerate(chunks):
                # Initialize a conversation with OpenAI for this chunk
                for q_index, q in enumerate(self.questions.keys()):
                    
                    # Generate the conversation to submit
                    conversation = self.generate_submission(chunk, q)
                    
                    # Use a while loop to allow 3 submission attempts
                    retry_count = 0
                    while retry_count < 3:
                        try:
                            answer, tokens_used = self.get_response_from_openai(conversation)
                            
                            # Store the answer and the corresponding chunk
                            if answer.lower() in acceptable_case_answers:
                                results_dict['case_report'].append(chunk)
                            else:
                                results_dict['other'].append(chunk)
                            
                            time.sleep(0.1)
                            break  # Exit the loop if successful
                        
                        # Handle Exceptions
                        except Exception as e:
                            retry_count, sleep_time = self.handle_response_exception(e, q_index, retry_count)
                            time.sleep(sleep_time)
                
                    if retry_count == 3:
                        results_dict['other'].append(f"Unidentified: chunk_{chunk_index+1}")
        # Join strings together and return the completed results
        results_dict['case_report'] = ' '.join(results_dict['case_report'])
        results_dict['other'] = ' '.join(results_dict['other'])
        return results_dict


