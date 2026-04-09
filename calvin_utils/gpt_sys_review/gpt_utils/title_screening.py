import pandas as pd
from tqdm import tqdm
from calvin_utils.gpt_sys_review.gpt_utils.openai_chat_base import OpenAIChatBase

class TitleScreener(OpenAIChatBase):
    """
    A class used to screen titles within a CSV file using OpenAI's chat models.
    This class extends the OpenAIChatBase, which provides the core functionality 
    to evaluate text using OpenAI's models. The TitleScreener class
    introduces methods specific to reading titles from a CSV and evaluating them.

    Attributes:
    ----------
    df : DataFrame
        A pandas DataFrame containing data from the provided CSV file.

    Methods:
    -------
    keyword_screen(keywords: List[str]) -> None:
        Screens titles in the DataFrame using a list of provided keywords.
    openai_screen() -> None:
        Evaluates titles in the DataFrame using OpenAI's chat models.
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
        
    Example Call:
    _____________
    # Define your API key, path to the CSV file, and the question for evaluation
    API_KEY = "YOUR_OPENAI_API_KEY"
    CSV_PATH = "path/to/your/titles.csv"
    EVALUATION_QUESTION = "Is this title related to medical research?"

    # Create an instance of the class
    title_screening = TitleScreenAPIRevised(api_key=API_KEY, csv_path=CSV_PATH, question=EVALUATION_QUESTION)

    # Perform keyword screening on titles
    keywords_list = ["focal", "lesion", "brain", "death", "case"]
    title_screening.keyword_screen(keywords=keywords_list)

    # Evaluate titles using OpenAI
    title_screening.openai_screen()

    # Save the screened titles to a new CSV file
    OUTPUT_PATH = "path/to/save/screened_titles.csv"
    title_screening.to_csv(output_path=OUTPUT_PATH)
    """
    def __init__(self, api_key_path, csv_path, question, model_choice="gpt3_small", keywords=None):
        self.csv_path = csv_path
        self.keywords = keywords
        self.question = question
        super().__init__(api_key_path=api_key_path, question_type="title_screening", model_choice=model_choice)
        self.df = pd.read_csv(csv_path)
    
    def keyword_screen(self):
        """
        Screen titles using a basic language model approach.
        
        Keywords = list of strings which can be used to identify relevant articles
        """
        if self.keywords:
            self.df["Keyword_Screen"] = self.df["Title"].apply(lambda title: int(any(word in title.lower() for word in self.keywords)))
    
    def launch_openai_evaluation(self, title):
        conversation = [
            {"role": "system", "content": "You are a helpful assistant. Responses should be (0 for No, 1 for Yes)"},
            {"role": "user", "content": f"Based on this title: {title}\n{self.question}\n\nResponse (0 for No, 1 for Yes)"}
            ]
        answer, tokens_used =  self.evaluate_with_openai(conversation)
        return 1 if "1" in answer else 0
        
    def openai_screen(self):
        """
        Screen titles using OpenAI GPT based on a posed question.
        """
        tqdm.pandas(desc="OpenAI Screening")
        self.df["OpenAI_Screen"] = self.df["Title"].progress_apply(lambda title: self.launch_openai_evaluation(title))
    
    def to_csv(self, output_path=None):
        """
        Save the dataframe with the screening results to a CSV.
        """
        if output_path is None:
            output_path = self.csv_path.split('.')[0] + "_cleaned.csv"
        self.df.to_csv(output_path, index=False)
        print(f"Saved finalized CSV to {output_path}")
        return output_path
        
    def run(self):
        """
        Orchestrator method
        """
        self.keyword_screen()
        self.openai_screen()
        output_path = self.to_csv()
