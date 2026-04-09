from calvin_utils.gpt_sys_review.gpt_utils.openai_json_evaluator import OpenAIJsonEvaluator

class CaseReportLabeler(OpenAIJsonEvaluator):
    """
    CaseReportLabeler labels and categorizes sections of text (e.g., medical case reports)
    using OpenAI's language models. It extends OpenAIJsonEvaluator to automate evaluation
    and labeling of text chunks based on user-defined questions and section headers.

    Attributes:
        text (str): The input text to be labeled and categorized.
        section_headers (dict): Dictionary of section headers used as keys for categorization.
        results_dict (dict): Dictionary to store categorized text chunks.
        questions (dict): Dictionary of questions used for labeling.
        verbose (bool): Whether to print verbose output.

    Methods:
        __init__(api_key_path, text, questions, section_headers, verbose=False):
            Initializes the CaseReportLabeler with API key, input text, questions, and section headers.
        _get_results_dict():
            Initializes and returns a results dictionary with keys from section_headers and empty lists as values.
        _chunk_text(text):
            Yields chunks of the input text using a chunking method.
        _evaluate_chunk(chunk):
            Yields (question, answer, tokens_used) for each question evaluated on a text chunk.
        _evaluate_response(answer, chunk, chunk_idx):
            Categorizes a chunk based on the answer received from the model.
        _finalize_results():
            Joins categorized text chunks into strings for each category.
        evaluate_all_files():
            Processes the entire text, evaluates each chunk, and categorizes them into results_dict based on model answers.
    """
    def __init__(self, api_key_path, text, questions, section_headers, question_token_estimate=500, is_azure=False, deployment_id=None, api_base=None, api_version=None, verbose=False):
        """
        Initialize the CaseReportLabeler.

        Args:
            api_key_path (str): Path to the API key file for authentication.
            text (str): The text to be processed and labeled.
            questions (dict): The questions or prompts to use for labeling.
            section_headers (dict): Dictionary of section headers relevant to the text.
            verbose (bool): Whether to print verbose output.
        """
        self.verbose = verbose
        self.text = text
        self.section_headers = section_headers
        self.text = self.text
        self.results_dict = self._get_results_dict()
        self.acceptable_answers = self._get_acceptable_answers()
        super().__init__(api_key_path, 
                         json_file_path=None, 
                         keys_to_consider=None,
                         question_token_estimate=question_token_estimate,
                         question_type="labelling", 
                         question=questions, 
                         answer_format='binary',
                         model_choice="gpt3_small_labeler",
                         is_azure=is_azure,
                         deployment_id=deployment_id,
                         api_base=api_base,
                         api_version=api_version,
                         debug=False, 
                         test_mode=False)

    def _get_acceptable_answers(self):
        """
        Returns a list of acceptable answers for labeling a chunk as positive.

        Returns:
            list: Acceptable affirmative answers.
        """
        return ["yes", "true", "y", "1"]
    
    def _get_results_dict(self):
        """
        Initializes the results dictionary with keys from section_headers and empty lists as values.
        Raises an error if more than one key is present, as this is not implemented yet.

        Returns:
            dict: Dictionary with keys from section_headers and empty lists as values.

        Raises:
            NotImplementedError: If more than one section header is provided.
        """
        keys = list(self.section_headers.keys())
        if len(keys) != 1:
            raise NotImplementedError("Handling multiple section headers is not implemented yet.")
        results_dict = {keys[0]: []}
        return results_dict

    def read_json(self, _):
        """
        Overrides the read_json method in the parent class.
        Returns an empty dictionary as JSON reading is not used here.

        Returns:
            dict: An empty dictionary.
        """
        return {}
    
    def _chunk_text(self, text):
        """
        Yields chunks of text using the chunker.

        Args:
            text (str): The text to be chunked.

        Yields:
            str: Chunks of the input text.
        """
        for chunk in self.call_chunker(text):
            yield chunk

    def _evaluate_chunk(self, chunk):
        """
        Yields (question, answer, tokens_used) for each question on the chunk.

        Args:
            chunk (str): The text chunk to evaluate.

        Yields:
            tuple: (question, answer, tokens_used) for each question.
        """
        for q_index, q in enumerate(self.questions.keys()):
            conversation = self.generate_submission(chunk, q)
            answer, tokens_used, retry_count = self.evaluate_with_openai(conversation, [q])
            yield q, answer, tokens_used
    
    def _evaluate_response(self, answer, chunk, chunk_idx):
        """
        Adds the chunk to the results_dict based on the answer.

        Args:
            answer (str): The model's answer.
            chunk (str): The text chunk.
            chunk_idx (int): The index of the chunk.
        """
        first_key = next(iter(self.results_dict))
        if any(ans in answer.lower() for ans in self.acceptable_answers):
            self.results_dict[first_key].append(chunk)
        else:
            if 'other' not in self.results_dict:
                self.results_dict['other'] = []
            self.results_dict['other'].append(chunk)
        if self.verbose: print(f"Evaluating chunk {chunk_idx} with answer: {answer}")
    
    def _finalize_results(self):
        """
        Finalizes the results by joining the strings in the results_dict.
        This method is called at the end of the evaluation process.
        """
        for key in self.results_dict:
            self.results_dict[key] = ' '.join(self.results_dict[key])

    def evaluate_all_files(self):
        """
        Evaluates the text to categorize text chunks based on the answers to questions.

        Returns:
            dict: Dictionary containing text chunks categorized under keys from section_headers.
        """
        if self.verbose: print("Evaluating text: ", self.text)
        for chunk in self._chunk_text(self.text):
            for chunk_idx, answer, tokens_used in self._evaluate_chunk(chunk):
                self._evaluate_response(answer, chunk, chunk_idx)
        self._finalize_results()
        return self.results_dict