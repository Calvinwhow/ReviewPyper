import os
import re
import cv2
import time
import textract
import subprocess
import numpy as np
import pytesseract
import pandas as pd
from tqdm import tqdm
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from pypaperretriever import PyPaperRetriever

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
        Recursively iterates through a directory and subdirectories of PDF files and extracts text using OCR,
        respecting a page threshold.

        Parameters
        ----------
        pdf_dir : str
            The directory containing the PDF files.
        output_dir : str
            The directory to save the extracted text to.
        page_threshold : int
            The maximum number of pages in a PDF for it to be processed.

        Returns
        -------
        Output directory path
        """
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(pdf_dir), 'pdf_txt')
        os.makedirs(output_dir, exist_ok=True)
        
        for file_name in tqdm(os.listdir(pdf_dir), desc="Processing PDFs"):
            full_path = os.path.join(pdf_dir, file_name)
            
            if os.path.isdir(full_path):
                # It's a directory; recurse into it
                # Calculate a specific output directory for this subdirectory
                sub_output_dir = os.path.join(output_dir, file_name)
                OCROperator.extract_text_from_pdf_dir(full_path, sub_output_dir, page_threshold)
            elif file_name.endswith('.pdf'):
                # It's a PDF file; process it
                input_file_path = full_path
                
                # Exclude if over page count
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
        Recursively iterates through a directory and subdirectories of PDF files, extracts text,
        and saves it to text files in an output directory.

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
            full_path = os.path.join(pdf_dir, file_name)
            
            if os.path.isdir(full_path):
                # It's a directory; recurse into it
                PDFTextExtractor.extract_text_from_pdf_dir(full_path, output_dir)
            elif file_name.endswith(".pdf"):
                # It's a PDF file; extract the text
                input_file_path = full_path
                output_file_name = f"{os.path.splitext(file_name)[0]}.txt"
                output_file_path = os.path.join(output_dir, output_file_name)
                text = PDFTextExtractor.extract_text_from_pdf(input_file_path)
                PDFTextExtractor.save_text_to_file(text, output_file_path)


class BulkPDFDownloader:
    """
    A class to bulk download PDFs for a list of DOIs from a CSV file.

    Attributes:
        csv_path (str): Path to the CSV file containing DOIs and screening info.
        directory (str): Directory to save PDFs.
        column (str): The column name used for filtering rows, default is "openai_screen_abstract".
    """
    def __init__(self, csv_path, email, allow_scihub=True, column=None):
        self.csv_path = csv_path
        self.master_df = pd.read_csv(self.csv_path)
        self.master_df.columns = [col.lower() for col in self.master_df.columns]
        self.directory = os.path.dirname(self.csv_path)
        self.pdf_dir_path = os.path.join(self.directory, 'PDFs')
        os.makedirs(self.pdf_dir_path, exist_ok=True)
        self.positive_abstract = column.lower() if column is not None else "openai_screen_abstract"
        self.email = email
        self.allow_scihub = allow_scihub

    def download_pdf(self, doi, pmid):
        """Helper function to download a PDF using PyPaperRetriever."""
        filename = f"{pmid}.pdf"
        retriever = PyPaperRetriever(email=self.email, doi=doi, download_directory=self.pdf_dir_path, allow_scihub=self.allow_scihub, filename=filename)
        result = retriever.find_and_download()
        return result.is_downloaded, result.filepath

    def run(self):
        filtered_df = self.master_df[self.master_df[self.positive_abstract] == 1]
        for index, row in filtered_df.iterrows():
            doi = row['doi']
            pmid = row['pmid']
            success, filepath = self.download_pdf(doi, pmid)
            if success:
                self.master_df.loc[self.master_df['doi'] == doi, 'pdf_downloaded'] = 1
                self.master_df.loc[self.master_df['doi'] == doi, 'pdf_path'] = filepath
            else:
                self.master_df.loc[self.master_df['doi'] == doi, 'pdf_downloaded'] = 0

        self.master_df.to_csv(self.csv_path, index=False)

class PdfPostProcess:
    """
    A class for post-processing downloaded PDFs, including updating the master list with download statuses and paths.

    Attributes:
        master_list_path (str): Path to the master CSV file containing metadata and download statuses.
    """
    def __init__(self, master_list_path):
        self.master_list_path = master_list_path
        self.directory = os.path.dirname(self.master_list_path)
        self.master_df = pd.read_csv(self.master_list_path)
        self.master_df.columns = [col.lower() for col in self.master_df.columns]

    def update_master_list(self):
        """Update the master list with download statuses and paths."""
        df_master = self.master_df
        if 'pdf_downloaded' not in df_master.columns:
            df_master['pdf_downloaded'] = 0
        if 'pdf_path' not in df_master.columns:
            df_master['pdf_path'] = ''

        pdf_dir_path = os.path.join(self.directory, 'PDFs')
        for index, row in df_master.iterrows():
            pmid = row['pmid']
            pdf_name = f"{pmid}.pdf"
            pdf_path = os.path.join(pdf_dir_path, pdf_name)

            if os.path.exists(pdf_path):
                df_master.loc[index, 'pdf_path'] = pdf_path
                df_master.loc[index, 'pdf_downloaded'] = 1

        df_master.to_csv(os.path.join(self.directory, 'master_list.csv'), index=False)

    def normalize_title(self, title):
        """Normalize title by removing non-alphabetic characters."""
        return re.sub('[^a-zA-Z]', '', title).lower()
    
    def run(self):
        """Orchestration method"""
        self.update_master_list()

class PDFTextExtractionWrapper:
    """
    A wrapper class for PDF text extraction.
    Uses both PDFTextExtractor and OCROperator classes for text extraction.
    Determines if a PDF is text-based or image-based and extracts text accordingly.
    """

    @staticmethod
    def is_text_based_pdf(file_path: str) -> bool:
        """
        Determines if a PDF is text-based by attempting to extract text from the first page.

        Parameters
        ----------
        file_path : str
            The path of the PDF file.

        Returns
        -------
        bool
            True if the PDF is text-based, False otherwise.
        """
        try:
            pdf_reader = PdfReader(file_path)
            first_page = pdf_reader.pages[0]
            text = first_page.extract_text()
            return bool(text)
        except Exception as e:
            print(f"Error checking if PDF is text-based: {e}")
            return False

    @staticmethod
    def extract_text(file_path: str, output_file_path: str) -> None:
        """
        Extracts text from a given PDF file using either direct extraction or OCR,
        depending on the PDF type (text-based or image-based).

        Parameters
        ----------
        file_path : str
            The path of the PDF file.
        output_file_path : str
            The path to save the extracted text.

        Returns
        -------
        None
        """
        if PDFTextExtractionWrapper.is_text_based_pdf(file_path):
            text = PDFTextExtractor.extract_text_from_pdf(file_path)
            PDFTextExtractor.save_text_to_file(text, output_file_path)
        else:
            text = OCROperator.extract_text_from_pdf(file_path)
            OCROperator.save_text_to_file(text, output_file_path)
