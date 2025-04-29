import pandas as pd
from tqdm import tqdm
from calvin_utils.gpt_sys_review.gpt_utils.title_screening import TitleScreener

class AbstractScreener(TitleScreener):
    """
    A class to evaluate abstracts from a CSV using the OpenAI API based on a posed question.

    Methods:
    -------
    keyword_screen(keywords: List[str]) -> None:
        Screens astracts in the DataFrame using a list of provided keywords.
    openai_screen() -> None:
        Evaluates astracts in the DataFrame using OpenAI's chat models.
    to_csv(output_path: str) -> None:
        Saves the DataFrame, including screening results, to a CSV file.

    Parameters:
    ----------
    api_key : str
        The API key for OpenAI.
    csv_path : str
        Path to the CSV file containing the titles to be screened.
    question : str
        The question posed to the OpenAI model for title evaluation.
    model_choice : str, optional (default="gpt3_small")
        The choice of OpenAI model to use for evaluation. Options include "gpt3_small", "gpt3_large", and "gpt4".
    """
    def __init__(self, api_key_path, csv_path, question, model_choice="gpt3_small", keywords=None):
        """
        Initializes the AbstractEvaluatorDocumented class with the path to the API key and the CSV containing the abstracts.

        Parameters:
        - api_key_path (str): Path to the file containing the OpenAI API key.
        - csv_path (str): Path to the CSV containing the abstracts.
        """
        self.csv_path = csv_path
        self.keywords = keywords
        super().__init__(api_key_path=api_key_path, csv_path=csv_path, question=question, model_choice=model_choice)
        self.df = pd.read_csv(csv_path)

    def keyword_screen(self):
        """
        Screen titles using a basic language model approach.
        
        Keywords = list of strings which can be used to identify relevant articles
        """
        if self.keywords:
            self.df["Keyword_Screen_Abstract"] = self.df["Abstract"].apply(lambda text: int(any(word in text.lower() for word in self.keywords)))
    
    def launch_openai_evaluation(self, text):
        conversation = [
            {"role": "system", "content": "You are a helpful binary assistant, only able to speak in 1s or 0s. Responses should be: 0 for No, 1 for Yes"},
            {"role": "user", "content": f"Based on this abstract: {text}\n{self.question}\n\nRespond 0 for No, 1 for Yes."}
            ]
        return self.evaluate_with_openai(conversation)
        
    def openai_screen(self):
        """
        Screen titles using OpenAI GPT based on a posed question.
        """
        tqdm.pandas(desc="OpenAI Screening")
        self.df["OpenAI_Screen_Abstract"] = self.df["Abstract"].progress_apply(lambda text: self.launch_openai_evaluation(text))
    
    def run(self):
        """
        Orchestrator method
        """
        self.keyword_screen()
        self.openai_screen()
        output_path = self.to_csv()
        return output_path