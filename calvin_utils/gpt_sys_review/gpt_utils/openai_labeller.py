import time
from calvin_utils.gpt_sys_review.gpt_utils.openai_json_evaluator import OpenAIJsonEvaluator

class CaseReportLabeler(OpenAIJsonEvaluator):
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

    def read_json(self):
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
        
        for file_name, selected_text in self.relevant_text_by_file.items():
            chunks = self.call_chunker(selected_text)                           # Chunk text by token limits
            for chunk_index, chunk in enumerate(chunks):                        # Send a query for each chunk
                for q_index, q in enumerate(self.questions.keys()):             # Initialize a conversation with OpenAI for this chunk
                    conversation = self.generate_submission(chunk, q)           # Generate the conversation to submit
                    answer, tokens_used = self.evaluate_with_openai(conversation) # Evaluate the chunk with OpenAI
        
                    if answer.lower() in acceptable_case_answers:               # Store the answer and the corresponding chunk
                        results_dict['case_report'].append(chunk)
                    else:
                        results_dict['other'].append(chunk) 
                    
        # Join strings together and return the completed results
        results_dict['case_report'] = ' '.join(results_dict['case_report'])
        results_dict['other'] = ' '.join(results_dict['other'])
        return results_dict
