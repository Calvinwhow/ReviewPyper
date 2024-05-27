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
        
        pdf_paths = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
        
        OCROperator.pdf_text_extraction(pdf_paths=pdf_paths, output_dir=output_dir, page_threshold=page_threshold)
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
            raise ValueError("Error: output_dir requires the absolute path to the dierctory to save results to.")
        os.makedirs(output_dir, exist_ok=True)
        
        OCROperator.pdf_text_extraction(pdf_paths=pdf_paths, output_dir=output_dir, page_threshold=page_threshold)
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

        OCROperator.pdf_text_extraction(pdf_paths=pdf_paths, output_dir=output_dir, page_threshold=page_threshold)
        return output_dir

    @staticmethod
    def pdf_text_extraction(pdf_paths, output_dir, page_threshold):
        """
        Extracts text from multiple PDF files specified by pdf_paths and saves the extracted text to individual text files in the specified output_dir. Each output file is named after its corresponding PDF file with an "_OCR" suffix.

        Parameters:

        pdf_paths (list): A list of file paths to the PDF files from which text needs to be extracted.
        output_dir (str): The directory where the extracted text files will be saved.
        page_threshold (int): The maximum number of pages a PDF can have for it to be processed. PDFs with more pages than this threshold will be skipped.
        This function iterates over the list of PDF file paths, checks if the file meets the criteria (i.e., is a PDF and does not exceed the page threshold), and attempts to extract text using OCR if necessary. Files that already have a corresponding output file in the target directory or encounter errors during processing are skipped with a message printed to the console.

        The function is robust against errors during the page count retrieval and text extraction processes, skipping any problematic PDFs and continuing with the rest.

        Note: This function depends on the OCROperator class for the actual PDF page count retrieval, text extraction, and saving the extracted text to a file.
        """
        for file_path in tqdm(pdf_paths, desc='Extracting text from PDFs.'):
            file_name = os.path.basename(file_path)
            if file_name.endswith('.pdf'):
                output_file_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_OCR.txt")
                if os.path.exists(output_file_path):
                    continue
                
                # Only inlcude manuscripts below page threshold.
                try:
                    page_count = OCROperator.get_pdf_page_count(file_path)
                except Exception as e:
                    print("Skipping due to error getting page number: " + str(e))
                    continue
                if page_count > page_threshold:
                    print(f"Skipping {file_name} as it has {page_count} pages, exceeding the threshold of {page_threshold}.")
                    continue
                
                # Extract all Text in PDF.
                try:
                    text = OCROperator.extract_text_from_pdf(file_path)
                except Exception as e:
                    print("Skipping document. Error occurred while extracting text: ", e)
                    continue
        
                OCROperator.save_text_to_file(text, output_file_path)
                
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
    This may need to be run with a VPN.

    Attributes:
        csv_path (str): Path to the CSV file containing DOIs and screening info.
        directory (str): Directory to save PDFs.
        column (str): The column name used for filtering rows, default is "OpenAI_Screen_Abstract".
    
    Methods:
        run(): Execute the PDF download using PyPaperBot.
    """
    def __init__(self, csv_path, column=None):
        self.csv_path = csv_path
        self.master_df = pd.read_csv(self.csv_path)
        self.directory = os.path.dirname(self.csv_path)
        self.pdf_dir_path = os.path.join(self.directory, 'PDFs')
        os.makedirs(self.pdf_dir_path, exist_ok=True) 
        self.positive_abstract = column if column is not None else "OpenAI_Screen_Abstract"

    def prepare_pypaperbot_v1(self, doi_list=None):    
        '''
        Helper function to prepare files and directories for pypaperbot.
        '''   
        if doi_list is None:
            # Read the CSV and filter rows based on the specified column
            filtered_df = self.master_df[self.master_df[self.positive_abstract] == 1]
            self.dois = filtered_df['DOI'].tolist()
        else:
            self.dois = doi_list

        # Create a text file to store the DOIs
        with open(os.path.join(self.directory, 'dois.txt'), 'w') as f:
            for doi in self.dois:
                f.write(f"{doi}\n")
                
        # Prepare PyPaperbot Information
        self.doi_file_path = os.path.join(self.directory, 'dois.txt')
        
    def orchestrate_download_v1(self, mirror="https://sci-hub.st"):
        '''
        Run the PyPaperBot Download
        Mirrors:  https://sci-hub.st | https://sci-hub.do | .st | .se | .ru
        '''
        #Mirrors:
        #TODO: re-run the download on unsuccessfuly downloaded files using a new mirror. 
        command = f"python -m PyPaperBot --doi-file=\"{self.doi_file_path}\" --dwn-dir=\"{self.pdf_dir_path}\" --scihub-mirror=\"{mirror}\""
        print("Running command:", command)

        try:
            subprocess.run(command, shell=True, text=True)
        except Exception as e:
            print("An error occurred:", e)
            
    def update_master_list(self):
        '''
        Will reference the result.csv generated by PyPaperBot to update the Master CSV. 
        Then, will intiate attempt to collect failed downloads from additional scihub mirrors.
        
        To ensure linked data between PyPaperBot (PPB) and ReviewPyPer, we use the DOI value shared between the two. 
        PPB requires a list of DOIs to download. For each DOI, it returns DOI, filename and success state.
        We then access our master list, index the row with the matching DOI, and update it based on PPB's results. 
        '''
        report_df = pd.read_csv(os.path.join(self.pdf_dir_path, 'result.csv'))
        
        # Prepare master df for update.
        if 'PDF_Downloaded' not in self.master_df.columns:
            self.master_df['PDF_Downloaded'] = ''
        if 'PDF_Path' not in self.master_df.columns:
                    self.master_df['PDF_Path'] = ''
        
        # Update Master list based on the PyPaperBot Results
        filtered_report_df = report_df[report_df["Name"].notna()]
        for index, row in filtered_report_df.iterrows():
            # Get values from the PyPaperBot result.csv file.
            doi = row["DOI"] # Will use this as the key to reference the master list. 
            name = row["Name"] # This is what PyPaperBot saved the give file as. 
            downloaded = row["Downloaded"] # PyPaperBot's indication of download success boolean
            filename = os.path.join(self.pdf_dir_path, str(name)) # where PyPaperBot saved to

            # Access the master list and update each item accordingly. 
            self.master_df.loc[self.master_df['DOI'] == doi, 'PDF_Downloaded'] = downloaded # Updating master list with PPB's bool
            self.master_df.loc[self.master_df['DOI'] == doi, 'PDF_Path'] = filename # Adding the path to the PDF downloaded by PPB.
            
        # Save the Updated Master List
        self.master_df.to_csv(self.csv_path, index=False)    
        
    def iteratively_download(self): 
        '''
        Iterate upon result_df multiple times until all mirrors have been attempted. 
        '''   
        
        mirror_list=["https://sci-hub.ru", "https://sci-hub.st", "https://sci-hub.se", "https://sci-hub.do"]    
        try:
            for mirror in mirror_list:
                report_df = pd.read_csv(os.path.join(self.pdf_dir_path, 'result.csv'))
                report_df = report_df[report_df["Name"].notna()] # Removing erroneous cells
                undownloaded_dois_list = report_df.loc[report_df["Downloaded"] == False, 'DOI'].tolist()
                print("----\n Found ", len(undownloaded_dois_list), " undownloaded documents. ")
                print("Attempting mirror: ", mirror, "\n ----") 
                if len(undownloaded_dois_list) > 0:
                    self.prepare_pypaperbot_v1(undownloaded_dois_list)
                    self.orchestrate_download_v1(mirror=mirror)
                    self.update_master_list()
                else:
                    pass
        except Exception as e:
            print("Error in iteration upon lists: %s" % e)
            
    def run(self):
        self.prepare_pypaperbot_v1()
        self.orchestrate_download_v1()
        self.update_master_list()
        self.iteratively_download()
        
class BulkPDFDownloaderV2(BulkPDFDownloader):
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
        try:
            retriever = PyPaperRetriever(email=self.email, doi=doi, download_directory=self.pdf_dir_path, allow_scihub=self.allow_scihub, filename=filename)
            result = retriever.find_and_download()
        except Exception as e:
            print(f"Error in PyPaperRetriever on PMID {pmid}: {e}")
            # Create a result object with default values in case of an error
            result = type('Result', (object,), {'is_downloaded': False, 'filepath': None})()
        return result.is_downloaded, result.filepath

    def run(self):
        filtered_df = self.master_df[self.master_df[self.positive_abstract] == 1]
        for index, row in tqdm(filtered_df.iterrows(), total=len(filtered_df)):
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
                
                df_master.loc[index, 'PDF_Path'] = pdf_path

        df_master.to_csv(os.path.join(self.directory, 'master_list.csv'), index=False)

    def update_master_list_v2(self):
        """Update the master list with download statuses and paths."""
        df_master = pd.read_csv(self.master_list_path)
        if 'PDF_Downloaded' not in df_master.columns:
            df_master['PDF_Downloaded'] = 0
        if 'PDF_Path' not in df_master.columns:
            df_master['PDF_Path'] = ''

        pdf_dir_path = os.path.join(self.directory, 'PDFs')
        for index, row in df_master.iterrows():
            pdf_name = f"{row['PMID']}.pdf"
            pdf_path = os.path.join(pdf_dir_path, pdf_name)
            # Add the path and update the downloaded column. 
            if  os.path.exists(pdf_path):
                df_master.loc[index, 'PDF_Path'] = pdf_path
                df_master.loc[index, 'PDF_Downloaded'] = True 
            else:
                # In this situation, we failed to download a positive hit.
                if row['OpenAI_Screen_Abstract'] == 1:
                    df_master.loc[index, 'PDF_Path'] = np.nan
                    df_master.loc[index, 'PDF_Downloaded'] = False
                # This article was not expected to be downloaded. Set blank. 
                else:
                    df_master.loc[index, 'PDF_Path'] = np.nan
                    df_master.loc[index, 'PDF_Downloaded'] = np.nan
                
        df_master.to_csv(os.path.join(self.directory, 'master_list.csv'), index=False)
        
    def normalize_title(self, title):
        """Normalize title by removing non-alphabetic characters."""
        return re.sub('[^a-zA-Z]', '', title).lower()
    
    def pmid_renaming(self, pdf_dir_path, df_master):
        unmatched_files = set()
        # List comprehension to filter only PDF files
        pdf_files = [f for f in os.listdir(pdf_dir_path) if f.endswith('.pdf')]
        for filename in tqdm(pdf_files, desc="First pass processing PDF files"):
            # Fix the filename to allow manipulation
            title_from_file = os.path.splitext(filename)[0]
            normalized_title_from_file = self.normalize_title(title_from_file)
                        
            # Only process incomplete files. 
            pmid_set = set(df_master['PMID'].astype(str))
            if title_from_file in pmid_set:
                # print("Object already complete. Skipping: ", title_from_file)
                continue
                
            # Iterate to match title
            file_matched = False
            for index, row in df_master.iterrows():
                title = row['Title']
                pmid = row['PMID']
                normalized_title = self.normalize_title(title)
                
                # Check if it matches. Update if it does, then move on. 
                if normalized_title_from_file == normalized_title:
                    new_filename = f"{pmid}.pdf"
                    os.rename(
                        os.path.join(pdf_dir_path, filename),
                        os.path.join(pdf_dir_path, new_filename)
                    )
                    df_master.loc[index, 'PDF_Path'] = os.path.join(pdf_dir_path, new_filename)
                    df_master.loc[index, 'PDF_Downloaded'] = 1
                    
                    # Added for legibility
                    file_matched = True
                    break
            # If it does not match, catalogue and move on. 
            if file_matched == False:
                unmatched_files.add(title_from_file)
            
        return df_master, unmatched_files
    
    def pmid_renaming_round_two(self, pdf_dir_path, df_master, unmatched_files, loose_match=False):
        unmatched_files2 = set()
        for title_from_file in tqdm(unmatched_files, desc=f"Collecting difficult to match files. Loose match {loose_match}."):
            # Skip blank files
            if title_from_file is None or title_from_file == 'none':
                continue 
            # Iterate to match titles
            file_matched = False
            for index, row in df_master.iterrows():
                title = row['Title']
                pmid = row['PMID']
                
                # Check if it matches. Update if it does, then move on. 
                if (title_from_file.lower() in title.lower()) or (title.lower() in title_from_file.lower()) \
                    or loose_match and self.normalize_title(title_from_file) in self.normalize_title(title) \
                        or (loose_match and self.normalize_title(title) in self.normalize_title(title_from_file)):
                    new_filename = f"{pmid}.pdf"
                    
                    # Do not overwrite existing files. Preceding files take precedence.
                    if os.path.exists(os.path.join(pdf_dir_path, new_filename)):
                        break
                    else:
                        os.rename(
                            os.path.join(pdf_dir_path, (title_from_file+'.pdf')),
                            os.path.join(pdf_dir_path, new_filename)
                        )
                        df_master.loc[index, 'PDF_Path'] = os.path.join(pdf_dir_path, new_filename)
                        df_master.loc[index, 'PDF_Downloaded'] = 1
                        
                        file_matched = True
                    break
            # If it does not match, catalogue and move on. 
            if file_matched == False:
                unmatched_files2.add(title_from_file)
        if loose_match:       
            print("Failed to match these files: \n" + "\n".join(unmatched_files2))
        return df_master, unmatched_files2

    def rename_pdfs_to_pmid(self):
        """Rename PDFs to PMIDs."""
        df_master = pd.read_csv(self.master_list_path)
        pdf_dir_path = os.path.join(self.directory, 'PDFs')
        df_master, unmatched_files = self.pmid_renaming(pdf_dir_path, df_master)
        df_master, unmatched_files = self.pmid_renaming_round_two(pdf_dir_path, df_master, unmatched_files)
        df_master, unmatched_files = self.pmid_renaming_round_two(pdf_dir_path, df_master, unmatched_files, loose_match=True)

        df_master.to_csv(os.path.join(self.directory, 'master_list.csv'), index=False)

    def run(self):
        """Orchestration method"""
        self.update_master_list()
        self.rename_pdfs_to_pmid()
        self.update_master_list_v2()
