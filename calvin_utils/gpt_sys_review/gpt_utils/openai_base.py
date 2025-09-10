import openai

class OpenAIBase:
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
    def __init__(self, api_key_path, is_azure=False, api_base=None, api_version=None):
        """
        Initializes the OpenAIEvaluator class.

        Parameters:
        - api_key_path (str): Path to the file containing the OpenAI API key.
        - article_type (str): The type of article (e.g., 'research', 'case').
        """
        self.api_key = self.read_api_key(api_key_path)
        self.is_azure = is_azure
        openai.api_key = self.api_key
        if self.is_azure:
            openai.api_type = "azure"
            openai.api_base = api_base
            openai.api_version = api_version

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
        