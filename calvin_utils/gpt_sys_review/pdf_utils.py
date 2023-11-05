import os
import re
import cv2
import textract
import subprocess
import numpy as np
import pytesseract
import pandas as pd
from tqdm import tqdm
from PyPDF2 import PdfReader
from pdf2image import convert_from_path

class OCROperator:
    """
    A class to handle OCR text extraction from PDFs in a directory.

    Attributes
    ----------
    None

    Methods
    -------
    extract_text_from_pdf(file_path: str) -> str:
        Extracts text from a given PDF file using OCR and returns it as a string.
    save_text_to_file(text: str, output_file_path: str) -> None:
        Saves the extracted text to a specified file path.
    extract_text_from_pdf_dir(pdf_dir: str, output_dir: str) -> None:
        Iterates through a directory of PDF files and extracts text using OCR.
    """
    
    @staticmethod
    def preprocess_image(image):
        """
        Preprocesses the image for OCR.

        Parameters
        ----------
        image : PIL.Image.Image
            The image to preprocess.

        Returns
        -------
        PIL.Image.Image
            The preprocessed image.
        """
        # Convert the image to grayscale
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Convert the image to binary (black and white)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
        
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """
        Extracts text from a PDF using OCR and returns it as a string.

        Parameters
        ----------
        file_path : str
            The path of the PDF file.

        Returns
        -------
        str
            The OCR-extracted text.
        """
        images = convert_from_path(file_path)
        text = ""

        for image in images:
            text += pytesseract.image_to_string(image)

        return text
    
    @staticmethod
    def get_pdf_page_count(file_path: str) -> int:
        pdf = PdfReader(file_path)
        return len(pdf.pages)

    @staticmethod
    def save_text_to_file(text: str, output_file_path: str) -> None:
        """
        Saves the extracted text to a specified file path.

        Parameters
        ----------
        text : str
            The text to save.
        output_file_path : str
            The file path to save the text to.

        Returns
        -------
        None
        """
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(text)

    @staticmethod
    def extract_text_from_pdf_dir(pdf_dir: str, output_dir: str = None, page_threshold: int = 50) -> None:
        """
        Iterates through a directory of PDF files and extracts text using OCR.

        Parameters
        ----------
        pdf_dir : str
            The directory containing the PDF files.
        output_dir : str
            The directory to save the extracted text to.

        Returns
        -------
        None
        """
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(pdf_dir), 'pdf_txt')
        os.makedirs(output_dir,exist_ok=True)
        for file_name in tqdm(os.listdir(pdf_dir)):
            if file_name.endswith('.pdf'):
                input_file_path = os.path.join(pdf_dir, file_name)
                
                #Exclude if over page count
                page_count = OCROperator.get_pdf_page_count(input_file_path)
                if page_count > page_threshold:
                    print(f"Skipping {file_name} as it has {page_count} pages, exceeding the threshold of {page_threshold}.")
                    continue
                
                output_file_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_OCR.txt")

                text = OCROperator.extract_text_from_pdf(input_file_path)
                OCROperator.save_text_to_file(text, output_file_path)
        return output_dir
                
    @staticmethod
    def extract_text_from_pdf_list(pdf_paths: list, output_dir: str = None, page_threshold=50) -> None:
        """
        Iterates through a list of PDF file paths and extracts text using OCR.

        Parameters
        ----------
        pdf_paths : list
            The list of PDF file paths.
        output_dir : str
            The directory to save the extracted text to.
        page_threshold : int, optional
            The maximum number of pages to process in a PDF file. Default is 50.

        Returns
        -------
        None
        
        """
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(pdf_dir), 'pdf_txt')
        os.makedirs(output_dir, exist_ok=True)
        
        for file_path in tqdm(pdf_paths):
            file_name = os.path.basename(file_path)
            if file_name.endswith('.pdf'):
                # Exclude if over page count
                page_count = OCROperator.get_pdf_page_count(file_path)
                if page_count > page_threshold:
                    print(f"Skipping {file_name} as it has {page_count} pages, exceeding the threshold of {page_threshold}.")
                    continue

                output_file_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_OCR.txt")

                text = OCROperator.extract_text_from_pdf(file_path)
                OCROperator.save_text_to_file(text, output_file_path)
        return output_dir
                
    @staticmethod
    def extract_text_from_master_list(master_list_path: str, output_dir: str = None, page_threshold=50) -> None:
        """
        Iterates through a master list CSV file and extracts text using OCR for each PDF path in 'PDF_Path' column.

        Parameters
        ----------
        master_list_path : str
            The path to the master list CSV file.
        output_dir : str, optional
            The directory to save the extracted text to.
        page_threshold : int, optional
            The maximum number of pages to process in a PDF file. Default is 50.

        Returns
        -------
        None
        """
        # Read the master list and get PDF paths
        df_master = pd.read_csv(master_list_path)
        pdf_paths = df_master['PDF_Path'].dropna().tolist()

        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(master_list_path), 'pdf_txt')
        os.makedirs(output_dir, exist_ok=True)

        for file_path in tqdm(pdf_paths):
            file_name = os.path.basename(file_path)
            if file_name.endswith('.pdf'):
                # Exclude if over page count
                page_count = OCROperator.get_pdf_page_count(file_path)
                if page_count > page_threshold:
                    print(f"Skipping {file_name} as it has {page_count} pages, exceeding the threshold of {page_threshold}.")
                    continue

                output_file_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_OCR.txt")

                text = OCROperator.extract_text_from_pdf(file_path)
                OCROperator.save_text_to_file(text, output_file_path)
        return output_dir

                
class PDFTextExtractor:
    """
    A class to handle PDF text extraction and saving it to a file.

    Attributes
    ----------
    None

    Methods
    -------
    extract_text_from_pdf(file_path: str) -> str:
        Extracts text from a given PDF file and returns it as a string.
    save_text_to_file(text: str, output_file_path: str) -> None:
        Saves a given text string to a specified file path.
    extract_text_from_pdf_dir(pdf_dir: str, output_dir: str) -> None:
        Iterates through a directory of PDF files, extracts text, and saves it to text files in an output directory.
    """
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """
        Extracts text from a given PDF file and returns it as a string.

        Parameters
        ----------
        file_path : str
            The path of the PDF file to extract text from.

        Returns
        -------
        str
            The extracted text as a string.
        """
        text = textract.process(file_path)
        return text.decode("utf-8")
    
    @staticmethod
    def save_text_to_file(text: str, output_file_path: str) -> None:
        """
        Saves a given text string to a specified file path.

        Parameters
        ----------
        text : str
            The text string to save.
        output_file_path : str
            The path where the text file will be saved.

        Returns
        -------
        None
        """
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write(text)
    
    @staticmethod
    def extract_text_from_pdf_dir(pdf_dir: str, output_dir: str) -> None:
        """
        Iterates through a directory of PDF files, extracts text, and saves it to text files in an output directory.

        Parameters
        ----------
        pdf_dir : str
            The directory containing the PDF files.
        output_dir : str
            The directory where text files will be saved.

        Returns
        -------
        None
        """
        for file_name in os.listdir(pdf_dir):
            if file_name.endswith(".pdf"):
                input_file_path = os.path.join(pdf_dir, file_name)
                output_file_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.txt")
                text = PDFTextExtractor.extract_text_from_pdf(input_file_path)
                PDFTextExtractor.save_text_to_file(text, output_file_path)

class BulkPDFDownloader:
    """
    A class to bulk download PDFs for a list of DOIs from a CSV file.

    Attributes:
        csv_path (str): Path to the CSV file containing DOIs and screening info.
        directory (str): Directory to save PDFs.
        column (str): The column name used for filtering rows, default is "OpenAI_Screen_Abstract".
    
    Methods:
        run(): Execute the PDF download using PyPaperBot.
    """
    def __init__(self, csv_path, column=None):
        self.csv_path = csv_path
        self.directory = os.path.dirname(self.csv_path)
        os.makedirs(self.directory, exist_ok=True)
        self.column = column if column is not None else "OpenAI_Screen_Abstract"

        # Read the CSV and filter rows based on the specified column
        df = pd.read_csv(self.csv_path)
        filtered_df = df[df[self.column] == 1]
        self.dois = filtered_df['DOI'].tolist()

        # Create a text file to store the DOIs
        with open(os.path.join(self.directory, 'input.txt'), 'w') as f:
            for doi in self.dois:
                f.write(f"{doi}\n")
                
    def update_master_list(self):
        """
        Update the master_list.csv to track which DOIs have been successfully downloaded.
        Adds a new column 'PDF_Downloaded' and sets it to 1 for successful downloads.
        """
        master_list_path = os.path.join(self.directory, 'master_list.csv')
        if os.path.exists(master_list_path):
            df_master = pd.read_csv(master_list_path)

            # Add 'PDF_Downloaded' column if it doesn't exist
            if 'PDF_Downloaded' not in df_master.columns:
                df_master['PDF_Downloaded'] = 0

            pdf_dir_path = os.path.join(self.directory, 'PDFs')

            for doi in self.dois:
                pdf_name = f"{doi.replace('/', '_')}.pdf"
                pdf_path = os.path.join(pdf_dir_path, pdf_name)
                
                if os.path.exists(pdf_path):
                    df_master.loc[df_master['DOI'] == doi, 'PDF_Downloaded'] = 1

            df_master.to_csv(os.path.join(master_list_path, ''), index=False)
            print("Updated 'master_list.csv' with PDF download status.")
        else:
            print("'master_list.csv' does not exist in the specified directory.")

    def run(self):
        doi_file_path = os.path.join(self.directory, 'input.txt')
        pdf_dir_path = os.path.join(self.directory, 'PDFs')
        os.makedirs(pdf_dir_path, exist_ok=True)
        
        # Run PyPaperBot
        command = f"python -m PyPaperBot --doi-file=\"{doi_file_path}\" --dwn-dir=\"{pdf_dir_path}\""
        subprocess.run(command, shell=True)
        
        self.update_master_list()

import os
import pandas as pd
import re

class PdfPostProcess:
    def __init__(self, master_list_path):
        self.master_list_path = master_list_path
        self.directory = os.path.dirname(self.master_list_path)

    def update_master_list(self):
        """Update the master list with download statuses and paths."""
        df_master = pd.read_csv(self.master_list_path)
        if 'PDF_Downloaded' not in df_master.columns:
            df_master['PDF_Downloaded'] = 0
        if 'PDF_Path' not in df_master.columns:
            df_master['PDF_Path'] = ''

        pdf_dir_path = os.path.join(self.directory, 'PDFs')
        for index, row in df_master.iterrows():
            title = self.normalize_title(row['Title'])
            pdf_name = f"{title}.pdf"
            pdf_path = os.path.join(pdf_dir_path, pdf_name)

            if os.path.exists(pdf_path):
                df_master.loc[index, 'PDF_Downloaded'] = 1
                df_master.loc[index, 'PDF_Path'] = pdf_path

        df_master.to_csv(os.path.join(self.directory, 'master_list_pdfs.csv'), index=False)

    def rename_pdfs_to_pmid(self):
        """Rename PDFs to PMIDs."""
        df_master = pd.read_csv(self.master_list_path)
        pdf_dir_path = os.path.join(self.directory, 'PDFs')
        
        for filename in os.listdir(pdf_dir_path):
            if filename.endswith('.pdf'):
                title_from_file = filename.replace('.pdf', '')
                normalized_title_from_file = self.normalize_title(title_from_file)

                for index, row in df_master.iterrows():
                    title = row['Title']
                    pmid = row['PMID']
                    normalized_title = self.normalize_title(title)

                    if normalized_title_from_file == normalized_title:
                        new_filename = f"{pmid}.pdf"
                        os.rename(
                            os.path.join(pdf_dir_path, filename),
                            os.path.join(pdf_dir_path, new_filename)
                        )
                        df_master.loc[index, 'PDF_Path'] = os.path.join(pdf_dir_path, new_filename)
                        break

        df_master.to_csv(os.path.join(self.directory, 'master_list_pdfs.csv'), index=False)

    def normalize_title(self, title):
        """Normalize title by removing non-alphabetic characters."""
        return re.sub('[^a-zA-Z]', '', title).lower()
    
    def run(self):
        """Orchestration method"""
        self.update_master_list()
        self.rename_pdfs_to_pmid()
