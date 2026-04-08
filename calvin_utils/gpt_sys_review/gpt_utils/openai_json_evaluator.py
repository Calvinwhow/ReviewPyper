import os 
import sys
import json
import time
from tqdm import tqdm
from calvin_utils.gpt_sys_review.gpt_utils.openai_chat_base import OpenAIChatBase

class OpenAIJsonEvaluator(OpenAIChatBase):
    def __init__(self, api_key_path, json_file_path, keys_to_consider, question, retain_chunks=False, question_token_estimate=500, question_type='research',  model_choice="gpt3_small", include_explanations=False,response_tokens=None, is_azure=False, deployment_id=None, api_base=None, api_version=None, debug=False, test_mode=True, max_workers=10):
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
        self.retain_chunks = retain_chunks
        if self.retain_chunks:
            self.chunk_dir = os.path.dirname(self.json_path) + "_chunks"
            os.makedirs(self.chunk_dir, exist_ok=True)
        else: 
            self.chunk_dir = None
        self.include_explanations = include_explanations
        self.json_data = self.read_json(json_file_path)
        self.get_model_data(model_choice)
        self.get_question_settings(question_type)
        
        if self.include_explanations:
            self.directive = "You are a medical assistant. Your task is to carefully evaluate the following medical record. Use both explicit information and reasonable inferences to answer the questions. Be as concise as possible."
        
        self.test_mode = test_mode
        self.max_workers = max_workers
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
    
    def save_chunks(self, file_name, chunks, metadata=None):

        base_save_file = os.path.join(self.chunk_dir, f'{file_name}_chunks.json')
        save_file = base_save_file
        count = 1
        if metadata:
            chunk_dict = {f'chunk_{i+1}': {'text': chunk, 'metadata': metadata[i]} for i, chunk in enumerate(chunks)}
        else:
            chunk_dict = {f'chunk_{i+1}': chunk for i, chunk in enumerate(chunks)}
        while os.path.exists(save_file):
            save_file = os.path.join(self.chunk_dir, f'{file_name}_chunks_{count}.json')
            count += 1
        with open(save_file, 'w') as f:
            json.dump(chunk_dict, f, indent=0)
        print(f"Saved to: {save_file}")
        return save_file


    ### Evlaluation Methods ###
 
    def evaluate_all_files(self):
        """Estimated cost: {tokens_used*self.cost*len(self.questions.items())*len(chunks)}')"""
        total_failed_chunks=0
        total_chunks=0
        total_retries=0
        try:
            total_tokens_used = 0
            if self.include_explanations:
                formatted_questions = (f'''For each of the following questions about the {self.chunk_flag} provided, '''
                f'''output an integer, followed by an explanation for your answer, followed by the exact DATE when the symptom first appeared (based on [REPORT DATE: YYYY-MM-DD] markers or other dates in the text). '''
                f'''If the symptom is completely unmentioned or not evaluated, respond 0 for the integer ("Unknown"). If the text explicitly states the patient DOES NOT have the symptom (e.g., normal gait), respond 1 for the integer ("No"). If the text explicitly confirms the patient DOES have the symptom, respond 2 for the integer ("Yes"). If the answer is 0 or 1, respond "Unknown" for the date. Do not try to make inferences. '''
                f'''All answers, explanations, and dates should be on one line, separated by the "|" character. '''
                f'''Suppose there are n questions. The format of your output should look like "integer_1|explanation_1|date_1|integer_2|explanation_2|date_2...integer_n|explanation_n|date_n". '''
                f'''For example: "2|The text mentions that the patient needs a cane to walk|2018-05-12|0|The text does not mention the heel-shin test|Unknown|1|The text explicitly notes normal ocular pursuit|Unknown" etc. '''
                f'''Be sure to answer every question separately and do not combine multiple questions into one answer. ''' 
                f'''Ignore any text which is part of a standardized questionnaire. The questions are:''')
                questions_w_explanations=[]                
                for i, question in enumerate(self.questions.keys()):

                    formatted_questions += f" | {question}"
                    questions_w_explanations += [question, 'EXPLANATION: '+question, 'Onset Date: '+question]

            else:
                questions_w_explanations=list(self.questions.keys())
                formatted_questions = (f'''For each of the following questions about the {self.chunk_flag} provided, '''
                f'''output a separate integer (0 for Unmentioned/Unknown, 1 for Explicit No, 2 for Yes), followed by the exact DATE when the symptom first appeared (based on [REPORT DATE: YYYY-MM-DD] markers or other dates in the text). '''
                f'''If the answer is 0 or 1, respond "Unknown" for the date. All answers and dates '''
                f'''should be on one line, separated by the "|" character. For example: "2|2018-05-12|1|Unknown|2|2019-01-01|0|Unknown" etc. Be sure to  '''
                f'''answer every question separately and do not combine multiple questions into one answer. ''' 
                f'''Ignore any text which is part of a standardized questionnaire. The questions are:''')
                formatted_questions += " ".join(questions_w_explanations)


            answers={}
            import concurrent.futures

            def process_chunk(chunk_tuple):
                file_name, chunk_index, chunk, chunk_metadata = chunk_tuple
                conversation = self.generate_submission(chunk, formatted_questions)
                answer, tokens_used, retries = self.evaluate_with_openai(conversation, questions_w_explanations)
                return file_name, chunk_index, answer, tokens_used, retries, chunk_metadata

            chunk_tasks = []
            for file_name, file_text in self.relevant_text_by_file.items():
                chunks, metadata = self.call_chunker(file_text)
                if self.retain_chunks:
                    self.save_chunks(file_name, chunks, metadata)
                answers[file_name] = {}
                for chunk_index, chunk in enumerate(chunks):
                    chunk_tasks.append((file_name, chunk_index, chunk, metadata[chunk_index]))
                    total_chunks += 1

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(process_chunk, task): task for task in chunk_tasks}
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(chunk_tasks), desc="Processing chunks"):
                    file_name, chunk_index, answer, tokens_used, retries, chunk_metadata = future.result()
                    total_tokens_used += tokens_used
                    total_retries += retries
                    if answer == "Unidentified":
                        total_failed_chunks += 1
                        answer_dict = {q: "Unidentified" for q in questions_w_explanations}
                    else:
                        answer_dict = dict(zip(questions_w_explanations, answer.split("|")))
                    answer_dict['metadata'] = chunk_metadata
                    answers[file_name][f"chunk_{chunk_index+1}"] = answer_dict
            
            print(f'Total tokens used: {total_tokens_used}. Estimated cost: {total_tokens_used*self.cost}')
            print(f'Total chunks: {total_chunks}. total number of retries: {total_retries}. Total failed chunks: {total_failed_chunks} ({total_failed_chunks/total_chunks*100:.1f}%)')
            with open('debug_answer.json', 'w') as f:
                json.dump(answers, f, indent=0)
            for record, mydict in answers.items():
                mydict = {key2: {key1: mydict[key1][key2] for key1 in answers[record]} for key2 in answers[record][next(iter(answers[record]))]}
                self.all_answers[record] = mydict
            return self.all_answers

        except KeyboardInterrupt:
            print("KeyboardInterrupt detected. Saving preliminary results to JSON and closing.")
            with open('debug_answer.json', 'w') as f:
                json.dump(answers, f, indent=0)
            sys.exit(0)
        except Exception as e:
            with open('debug_answer.json', 'w') as f:
                json.dump(answers, f, indent=0)
            raise RuntimeError(f"Critical error occured: \n\t{e}. Saving preliminary results and aborting.")
        