import os 
import sys
import json
import time
from tqdm import tqdm
from calvin_utils.gpt_sys_review.gpt_utils.openai_chat_base import OpenAIChatBase

class OpenAIJsonEvaluator(OpenAIChatBase):
    def __init__(self, api_key_path, json_file_path, keys_to_consider, question, question_token_estimate=500, question_type='research',  model_choice="gpt3_small", debug=False, test_mode=True):
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
        super().__init__(api_key_path, question_token_estimate=question_token_estimate, question_type=question_type, model_choice=model_choice)
        self.json_path = json_file_path
        self.keys_to_consider = keys_to_consider
        self.all_answers = {}
        self.debug = debug
        self.questions = question
        self.json_data = self.read_json(json_file_path)
        self.get_model_data(model_choice)
        self.get_question_settings(question_type)
        
        self.test_mode = test_mode
        if self.test_mode and self.json_data:
            first_key = next(iter(self.json_data.keys()))
            self.json_data = {first_key: self.json_data[first_key]}
            print(f'Will evaluate only {len(self.json_data)} articles for testing.')
        self.extract_relevant_text()
    
    ### JSON handling ###
    
    def read_json(self, json_file_path):
        """Reads JSON data from a file and returns it as a dictionary."""
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
        """Extracts and stores relevant text sections based on keys_to_consider"""
        self.relevant_text_by_file = {}
        for file_name, sections in self.json_data.items():
            selected_text = ""
            for key, value in sections.items():
                if key in self.keys_to_consider:
                    selected_text += value
            self.relevant_text_by_file[file_name] = selected_text
            
    def save_to_json(self, output_dict):
        """Saves the labeled sections to a JSON file."""
        # Create a new directory in the same root folder
        out_dir = os.path.dirname(self.json_path) + "_evaluated"
        os.makedirs(out_dir, exist_ok=True)
        base_save_file = os.path.join(out_dir, f'{self.question_type}_evaluations.json')
        save_file = base_save_file
        count = 1
        while os.path.exists(save_file):
            save_file = os.path.join(out_dir, f'{self.question_type}_evaluations_{count}.json')
            count += 1
        with open(save_file, 'w') as f:
            json.dump(output_dict, f, indent=0)
        print(f"Saved to: {save_file}")
        return save_file
    
    ### Evlaluation Methods ###
 
    def evaluate_all_files(self):
        """Estimated cost: {tokens_used*self.cost*len(self.questions.items())*len(chunks)}')"""
        try:
            for file_name, file_text in tqdm(self.relevant_text_by_file.items()):
                chunks = self.call_chunker(file_text)  # Chunk text by token limits
                self.all_answers[file_name] = {}            # Initialize a dictionary to store chunk-level answers for each question
                for question in self.questions.keys():
                    self.all_answers[file_name][question] = {}

                for chunk_index, chunk in enumerate(chunks):     # Send a query for each chunk
                    for q_index, q in enumerate(self.questions.keys()): # Initialize a conversation with OpenAI for this chunk
                        conversation = self.generate_submission(chunk, q)   # Generate the conversation to submit
                        answer, tokens_used = self.evaluate_with_openai(conversation) # Evaluate the chunk with OpenAI
                        self.all_answers[file_name][q][f"chunk_{chunk_index+1}"] = answer       # Store the answer for this question and this chunk
            return self.all_answers
        
        except KeyboardInterrupt:
            print("KeyboardInterrupt detected. Saving results to JSON and closing.")
            self.save_to_json(self.all_answers)
            sys.exit(0)
        except Exception as e:
            self.save_to_json(self.all_answers)
            raise RuntimeError(f"Critical error occured: \n\t{e}. Saving results and aborting.")