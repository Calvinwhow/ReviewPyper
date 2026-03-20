import os 
import sys
import json
import time
from tqdm import tqdm
from calvin_utils.gpt_sys_review.gpt_utils.openai_chat_base import OpenAIChatBase

class OpenAIJsonEvaluator(OpenAIChatBase):
    def __init__(self, api_key_path, json_file_path, keys_to_consider, question, answer_format, retain_chunks=False, question_token_estimate=500, question_type='research',  model_choice="gpt3_small", response_tokens=None, is_azure=False, deployment_id=None, api_base=None, api_version=None, debug=False, test_mode=True, test_count=5, max_workers=10):
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
        if response_tokens is None:
            response_tokens = len(question)*50
        super().__init__(api_key_path, question_token_estimate=question_token_estimate, question_type=question_type, model_choice=model_choice, response_tokens=response_tokens, is_azure=is_azure, deployment_id=deployment_id, api_base=api_base, api_version=api_version, debug=debug)
        self.json_path = json_file_path
        self.keys_to_consider = keys_to_consider
        self.all_answers = {}
        self.questions = question
        self.answer_format = answer_format
        self.retain_chunks = retain_chunks
        if self.retain_chunks:
            self.chunk_dir = os.path.dirname(self.json_path) + "_chunks"
            os.makedirs(self.chunk_dir, exist_ok=True)
        else: 
            self.chunk_dir = None
        # self.include_explanations = include_explanations
        self.json_data = self.read_json(json_file_path)
        self.get_model_data(model_choice)
        self.get_question_settings(question_type)
        
        self.test_mode = test_mode
        self.max_workers = max_workers
        if self.test_mode and self.json_data:
            self.json_data = {key:val for key, val in list(self.json_data.items())[:test_count]}
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
    
    def save_chunks(self, file_name, chunks):

        base_save_file = os.path.join(self.chunk_dir, f'{file_name}_chunks.json')
        save_file = base_save_file
        count = 1
        chunk_dict = {f'chunk_{i+1}': chunk for i, chunk in enumerate(chunks)}
        while os.path.exists(save_file):
            save_file = os.path.join(self.chunk_dir, f'{file_name}_chunks_{count}.json')
            count += 1
        with open(save_file, 'w') as f:
            json.dump(chunk_dict, f, indent=0)
        print(f"Saved to: {save_file}")
        return save_file


    ### Evlaluation Methods ###
    def evaluate_single_file(self, file_name, file_text, formatted_questions, questions_w_explanations):

        print(' Evaluating '+file_name)
        chunks = self.call_chunker(file_text)  # Chunk text by token limits

        if self.retain_chunks:
            chunk_path=self.save_chunks(file_name, chunks)

        file_answers = {} # Initialize a dictionary to store chunk-level answers for each question
        file_retries=0
        file_tokens_used=0
        file_failed_chunks=0

        for chunk_index, chunk in enumerate(chunks):     # Send a query for each chunk

            conversation = self.generate_submission(chunk, formatted_questions)   # Generate the conversation to submit
            answer, tokens_used, retries = self.evaluate_with_openai(conversation, questions_w_explanations) # Evaluate the chunk with OpenAI
            file_tokens_used += tokens_used
            file_retries += retries

            if answer=="Unidentified":
                file_failed_chunks+=1
                chunk_answers={q:"Unidentified" for q in questions_w_explanations}
            else:
                chunk_answers=dict(zip(questions_w_explanations,answer.split("|")))  # Convert the answer string to a dictionary

            file_answers[f"chunk_{chunk_index+1}"] = chunk_answers      # Store the answer for this question and this chunk
        
        return file_answers, file_tokens_used, file_retries, file_failed_chunks
    
    def format_questions(self, questions_list):
        """Formats the questions for submission to the OpenAI API."""
        if self.answer_format in ['inclusion', "binary_with_explanations",'binary_with_unknown_and_explanations',"severity_with_explanations",]:
            questions_w_explanations=[prepend+q for q in questions_list for prepend in ['','EXPLANATION: ']]     
        elif self.answer_format=="binary" or self.answer_format=="binary_without_explanations":
            questions_w_explanations=questions_list
        else:            
            myerror=f'answer_format {self.answer_format} is invalid. Allowed answer types are "inclusion", "binary_without_explanations", "binary_with_explanations", "binary_with_unknown_and_explanations", "severity_with_explanations"'
            raise ValueError(myerror)
        
        formatted_questions = json.load(open(os.path.join(os.path.dirname(__file__), 'prompts.json')))[self.answer_format]
        formatted_questions = formatted_questions.replace("[CHUNK_FLAG]", self.chunk_flag)
        formatted_questions += " ".join(questions_w_explanations)
        
        return formatted_questions, questions_w_explanations
    

    def swap_heirarchy(self, answers_dict):
        """Swaps the hierarchy of the answers dictionary from 
        {file: {chunk: {question: answer}}} to {file: {question: {chunk: answer}}}"""
        reorganized_answers = {}
        for record, mydict in answers_dict.items():

            mydict = {key2: {key1: mydict[key1][key2] for key1 in answers_dict[record]} for key2 in answers_dict[record][next(iter(answers_dict[record]))]}
            reorganized_answers[record] = mydict

        return reorganized_answers


    def evaluate_all_files(self):
        """Estimated cost: {tokens_used*self.cost*len(self.questions.items())*len(chunks)}')"""

        try:
            total_tokens_used = 0

            formatted_questions, questions_w_explanations = self.format_questions(list(self.questions.keys()))

            answers_dict={}
            total_failed_chunks=0
            total_retries=0
            total_chunks=0
            for file_name, file_text in tqdm(self.relevant_text_by_file.items()):

                answers_dict[file_name], file_tokens_used, file_retries, file_failed_chunks=self.evaluate_single_file(file_name, file_text, formatted_questions, questions_w_explanations)

                total_tokens_used += file_tokens_used
                total_retries+=file_retries
                total_failed_chunks+=file_failed_chunks
                total_chunks+=len(answers_dict[file_name])

            # print(f'Total tokens used: {total_tokens_used}. Estimated cost: {total_tokens_used*self.cost}')
            print(f'Total chunks: {total_chunks}. total number of retries: {total_retries}. Total failed chunks: {total_failed_chunks} ({total_failed_chunks/total_chunks*100:.1f}%)')

            self.all_answers=self.swap_heirarchy(answers_dict)

            return self.all_answers

        except KeyboardInterrupt:
            print("KeyboardInterrupt detected. Saving preliminary results to JSON and closing.")
            with open('debug_answer.json', 'w') as f:
                json.dump(answers_dict, f, indent=0)
            sys.exit(0)
        except Exception as e:
            with open('debug_answer.json', 'w') as f:
                json.dump(answers_dict, f, indent=0)
            raise RuntimeError(f"Critical error occured: \n\t{e}. Saving preliminary results and aborting.")
        