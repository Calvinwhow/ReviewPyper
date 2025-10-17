import time
import openai
import numpy as np
from calvin_utils.gpt_sys_review.txt_utils import TextChunker
from calvin_utils.gpt_sys_review.gpt_utils.openai_base import OpenAIBase

class OpenAIChatBase(OpenAIBase):
    """
    Base class to evaluate text chunks using OpenAI's chat models.
    """
    
    def __init__(self, api_key_path, question_type, question_token_estimate=500, model_choice="gpt3_small", response_tokens=50, is_azure=False, deployment_id=None, api_base=None, api_version=None, debug=False):
        super().__init__(api_key_path, is_azure=is_azure, api_base=api_base, api_version=api_version)
        self.question_type = question_type
        self.debug= debug
        self.q_index = 0
        self.question_token_estimate=question_token_estimate
        self.deployment_id=deployment_id
        self.is_azure = is_azure
        self.get_model_data(model_choice)
        if response_tokens > self.token_limit:
            print(f"Warning: desired response_tokens {response_tokens} exceeds token limit {self.token_limit}. Setting response_tokens to {self.token_limit}.")
            self.response_tokens = self.token_limit
        else:
            self.response_tokens = int(response_tokens)

    
    ### setter/getter methods ###
        
    def get_model_data(self, model_choice):
        """Sets values for the OpenAI model to use."""
        self.temperature = 1.0
        models = {
            "gpt4.1": {"name": "gpt-4.1", "token_limit": 32768, "cost": 0.03 / 1000},
            "gpt4": {"name": "gpt-4", "token_limit": 7000, "cost": 0.03 / 1000}, #actual limit is 8192
            "gpt4o-mini": {"name": "gpt-4o-mini", "token_limit": 10000, "cost": 0.00015 / 1000}, #actual limit is 128k, but that's impractical for testing.
            "gpt3_large": {"name": "gpt-3.5-turbo-16k", "token_limit": 16385, "cost": 0.003 / 1000},
            "gpt3_small": {"name": "gpt-3.5-turbo", "token_limit": 4097, "cost": 0.0015 / 1000},
            "gpt3_small_labeler": {"name": "gpt-3.5-turbo", "token_limit": 1000, "cost": 0.0015 / 1000}
        }
        if model_choice not in models:
            raise ValueError(f"Model choice {model_choice} not supported. Please choose from: {', '.join(models.keys())}.")
        
        self.model = models[model_choice]["name"]
        self.token_limit = (models[model_choice]["token_limit"] - np.round(1.2*(self.question_token_estimate)))//2
        self.cost = models[model_choice]["cost"]
        
    def get_question_settings(self, question_type):
        """
        Sets the manner in which directives and questions are posed to the model.
        """
        if self.question_type=="extraction":
            self.directive = "You are a research assistant. Your task is to carefully evaluate the following research report. Use both explicit information and reasonable inferences to answer the questions. Be as concise as possible."
            self.chunk_flag = "[RESEARCH REPORT]"
        elif self.question_type=="emr_extraction":
            self.directive = "You are a medical assistant. Your task is to carefully evaluate the following medical record. Use both explicit information and reasonable inferences to answer the questions. Be as concise as possible."
        elif self.question_type=="emr_strict_extraction":
            self.directive = "You are a medical assistant. Your task is to carefully evaluate the following medical record. Use only explicit information to answer the questions and do not make inferences. Be as concise as possible."
            self.chunk_flag = "[EMR REPORT]"
        elif self.question_type=="case":
            self.directive = "You are a medical assistant. Your task is to carefully evaluate the following case report. Use both explicit information and reasonable inferences to answer the questions. Be as concise as possible."
            self.chunk_flag = "[MEDICAL RECORD]"
        elif self.question_type=="summarizer":
            self.directive = "An LLM saw multiple chunks of a text file and answered the below question. What was the ultimate answer?"
            self.chunk_flag = "[LLM ANSWERS]"
        elif self.question_type=="inclusion":
            self.directive = "You are a helpful binary assistant, only able to speak in 1s or 0s. Your task is to carefully evaluate the following medical article. Use both explicit information and reasonable inferences to answer the questions. Responses should be: 0 for No, 1 for Y."
            self.chunk_flag = "[MEDICAL ARTICLE]"
        elif self.question_type=="labelling":
            self.directive = "You are a text labelling assistant. Your task is to carefully evaluate the following case report. Use both explicit information and reasonable inferences to answer the questions. Responses should be: 0 for No, 1 for Y."
            self.chunk_flag = "[SEGMENT]"
        else:
            raise ValueError(f"Model choice {question_type} not supported, please choose gpt4, gpt3_large, or gpt3_small.")
    
    ### Chunking methods ###
    # def add_context_to_chunks(self, chunks, debug=False):
    #     """Method to append a message to the end of every chunk. Set in self.get_question_settings"""
    #     if self.chunk_end is not None:
    #         for i in range(len(chunks)):
    #             chunks[i] += self.chunk_end
    #     print(chunks) if debug else None
    #     return chunks
        
    def call_chunker(self, selected_text):
        """
        Uses TextChunker defined in text_utils.py to extract text in chunks
        """
        self.text_chunker = TextChunker(selected_text, self.token_limit)
        self.text_chunker.chunk_text()
        chunks = self.text_chunker.get_chunks()
        # if self.debug:
        #         print('Text associated with file:', selected_text)
        #         print(f'Allowing {self.token_limit} tokens per submission')
        #         print('Number of chunks:', len(chunks))
        # chunks = self.add_context_to_chunks(chunks)
        return chunks
    
    def generate_submission(self, chunk, question):
        """
        Prepares the submission to the OpenAI Model
        """
        conversation = [{"role": "system", "content": f"{self.directive}"},
                        {"role": "user", "content": f"{self.chunk_flag}: {chunk}"}]
        conversation.append({"role": "user", "content": question})
        return conversation
    
    ### Methods for interfacing with openai ###
    def evaluate_with_openai(self, conversation, questions_list):
        """Evaluates the title using OpenAI GPT."""
        retry_count = 0
        while retry_count < 4:
            try:
                tokens_used=0 # if there's an error in get_response_from_openai, tokens_used won't be defined and can't be returned, so set it to 0 here.
                answer, tokens_used = self.get_response_from_openai(conversation)
                
                is_answer_good, answer = self.verify_response_formatting(answer,questions_list)

                if not is_answer_good:
                    with open('error_log.txt', 'a') as f:
                        answer_count=len(answer.replace('\n','|').replace('||','|').split('|'))
                        f.write(f"{answer_count} {answer}\n\n")
                    raise IndexError("ChatGPT response does not have the correct number of answers")
                
                self.q_index += 1
                return answer, tokens_used, retry_count
            
            except Exception as e:
                retry_count, sleep_time = self.handle_response_exception(e, retry_count)
                time.sleep(sleep_time)

        print("Failed to get a response after 4 attempts. Setting chunk to Unidentified")
        return "Unidentified", tokens_used, retry_count

    def get_response_from_openai(self, conversation):
        """Sends a conversation to OpenAI and retrieves the assistant's last answer."""
        if self.is_azure:
            response = openai.ChatCompletion.create(
                deployment_id=self.deployment_id,
                model=self.model,
                messages=conversation,
                temperature=self.temperature,
                max_tokens=int(self.response_tokens)
            )
        else:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=conversation,
                temperature=self.temperature,
                max_tokens=int(self.response_tokens)
            )
        if self.debug:
            with open('debug_openai_chat.txt', 'a') as f:
                f.write('Conversation: '+str(conversation)+'\n\n')
                f.write('Response: '+str(response['choices'][-1]['message']['content'])+'\n\n\n')

        return response['choices'][-1]['message']['content'], response["usage"]["total_tokens"]

    def verify_response_formatting(self, answer,questions):
        """Verifies that the response from ChatGPT has the correct formatting, i.e. there is an answer
        for each question and they are separated by a '|' character."""

        while answer[-1] in ["|",' ', '\n']:
            answer=answer[:-1]
        answer=answer.replace('||','|')

        if len(answer.split("|"))==len(questions):
            return True, answer
        
        new_answer=answer.replace('\n','|')
        new_answer=new_answer.replace('||','|')

        if len(new_answer.split("|"))==len(questions):
            return True, new_answer
    
        return False, answer

    def handle_response_exception(self, e, retry_count, q_index=0):
        """Handles exceptions during API calls to OpenAI"""
        if type(e).__name__ == 'RateLimitError':
            print(f"Rate limit error: {e}. Retrying submission. Attempt:({retry_count+1})")
            return retry_count + 1, 30
        elif  type(e).__name__ == 'IndexError':
            print(f"Index error: {e}. Retrying submission. Attempt:({retry_count+1})")
            return retry_count + 1, 5
        else:
            print(f"An error occurred: {e}. Retrying submission. Attempt:({retry_count+1})")
            return retry_count + 1, 1
