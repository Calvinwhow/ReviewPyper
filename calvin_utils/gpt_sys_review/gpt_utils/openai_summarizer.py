from calvin_utils.gpt_sys_review.gpt_utils.openai_json_evaluator import OpenAIJsonEvaluator

class OpenAISummarizer(OpenAIJsonEvaluator):
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
    def __init__(self, api_key_path, text, question, is_azure=False,deployment_id=None, api_base=None, api_version=None, verbose=False):
        """
        Initialize the CaseReportLabeler.

        Args:
            api_key_path (str): Path to the API key file for authentication.
            text (str): The text to be processed and labeled.
            questions (dict): The questions or prompts to use for labeling.
            verbose (bool): Whether to print verbose output.
        """
        self.verbose = verbose
        self.question = question
        self.text = text
        super().__init__(api_key_path, 
                         json_file_path=None, 
                         keys_to_consider=None, 
                         question_type="summarizer", 
                         question=question, 
                         model_choice="gpt4",
                         is_azure=is_azure, deployment_id=deployment_id, api_base=api_base, api_version=api_version,
                         debug=False, 
                         test_mode=False)

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

    def _evaluate_single_chunk(self, chunk):
        """
        Yields (question, answer, tokens_used) for each question on the chunk.

        Args:
            chunk (str): The text chunk to evaluate.

        Yields:
            tuple: (question, answer, tokens_used) for each question.
        """
        conversation = self.generate_submission(chunk, self.question)
        answer, tokens_used = self.evaluate_with_openai(conversation)
        return answer

    def evaluate_text(self):
        """
        Evaluates the text to categorize text chunks based on the answers to questions.

        Returns:
            dict: Dictionary containing text chunks categorized under keys from section_headers.
        """
        chunks = self.call_chunker(self.text)
        if len(chunks) > 1:
            raise ValueError("More than one chunk found. Only one chunk is allowed.")
        return self._evaluate_single_chunk(chunks[0])     #<-- only evaluate the first chunk
         