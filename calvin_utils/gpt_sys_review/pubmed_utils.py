import pandas as pd
from io import BytesIO
from Bio import Entrez
import requests
import subprocess
from glob import glob
import os
import requests
from lxml import etree
from itertools import chain
from tqdm import tqdm
import re
import numpy as np

tqdm.pandas()

class PubMedSearcher:
    """
    A class to search PubMed for articles based on a search string and retrieve article information.

    Attributes:
    - search_string (str): The search string to use when querying PubMed.
    - df (DataFrame): A DataFrame containing the retrieved articles.
    - email (str): The email address to use when querying PubMed.

    User-Facing Methods:
    - search: Searches PubMed for articles based on the search string and retrieves the specified number of articles.
    - download_articles: Downloads articles from the DataFrame to the specified directory (open access is prioritized, but may use PyPaperBot as a fallback).
    - fetch_references: Fetches references for each article in the DataFrame using multiple methods.
    - standardize_references: Standardizes the references column in the DataFrame to only contain the following keys: ['doi', 'pmid', 'pmcid', 'title', 'authors']
    - fetch_cited_by: Fetches list of articles that cite each article in the DataFrame using Europe PMC (only works for articles with a record in Europe PMC)
    - download_xml_fulltext: Downloads the XML full text for each article in the DataFrame to the specified directory (rarely available, but can be useful).
    - check_open_access: Checks if each article in the DataFrame is open access using Unpaywall and fills in the relevant columns ['is_oa', 'best_oa_location_url', 'pdf_url_1', 'pdf_url_2', 'pdf_url_3', 'pdf_url_4', 'europe_pmc_url']
    - save: Saves the df to a CSV file.
    """

    def __init__(self, search_string=None, df=None, email="lesion_bank@gmail.com"):
        """
        Initializes the PubMedSearcher object with a search string and an optional DataFrame.

        Parameters:
        search_string (str): The search string to use when querying PubMed.
        df (DataFrame): An optional DataFrame that may already contain articles, previous search results, etc.
        email (str): The email address to use when querying PubMed.
        """
        self.search_string = search_string
        self.df = df if df is not None else pd.DataFrame()
        if df is not None:
            self._validate_dataframe(df)
        elif search_string is not None:
            self.df = pd.DataFrame()
        self.email = email

    def search(self, count=10, min_date=None, max_date=None, order_by='chronological'):
        """
        Searches PubMed for articles based on the search string and retrieves the specified number of articles.

        Parameters:
        count (int): The number of articles to retrieve.
        min_date (int, optional): The minimum publication year to consider.
        max_date (int, optional): The maximum publication year to consider.
        order_by (str, optional): The order in which to retrieve articles. Can be 'chronological' or 'relevance'. Defaults to 'chronological'.
        """
        if not self.search_string:
            raise ValueError("Search string is not provided")

        Entrez.email = self.email
        search_params = {
            'db': "pubmed",
            'term': self.search_string,
            'retmax': count,
            'sort': 'relevance' if order_by == 'relevance' else 'pub date',
        }

        if min_date is not None:
            search_params['mindate'] = str(min_date)
        if max_date is not None:
            search_params['maxdate'] = str(max_date)

        search_handle = Entrez.esearch(**search_params)
        search_results = Entrez.read(search_handle)
        search_handle.close()

        id_list = search_results['IdList']
        fetch_handle = Entrez.efetch(db="pubmed", id=id_list, retmode="xml")
        records_xml_bytes = fetch_handle.read()
        fetch_handle.close()

        records_df = self._parse_records_to_df(records_xml_bytes)
        self.df = pd.concat([self.df, records_df], ignore_index=True)

    def download_articles(self, download_directory="PDFs", allow_pypaperbot=True, save_progress=True, max_downloads=None, enumerate=False):
        """
        Downloads articles from the DataFrame to the specified directory. Tries to download open access articles first, then uses PyPaperBot as a fallback.

        Parameters:
        download_directory (str): The directory where the PDF files should be saved.
        allow_pypaperbot (bool): Whether to use PyPaperBot as a fallback for downloading articles.
        save_progress (bool): Whether to save the DataFrame after downloading each article. Good when downloading large numbers of articles that may not complete in one go.
        max_downloads (int): The maximum number of articles to download. If None, all articles will be downloaded.
        enumerate (bool): Whether to enumerate the download directories based on the article index.
        """

        if self.df.empty:
            print("DataFrame is empty.")
            return

        if 'download_complete' not in self.df.columns:
            self.df['download_complete'] = 'Not started'
        if 'pdf_filepath' not in self.df.columns:
            self.df['pdf_filepath'] = None

        for index, row in tqdm(self.df.iterrows(), total=self.df.shape[0], desc="Downloading articles"):
            count_downloads = self.df['download_complete'].value_counts().get('Complete', 0)
            if max_downloads and count_downloads >= max_downloads:
                print(f"Maximum number of downloads reached ({max_downloads}).")
                break
            if row.get('download_complete') == 'Complete' or row.get('download_complete') == 'Unavailable':
                continue
            
            custom_download_dir = self._determine_download_directory(row, download_directory, index, enumerate)
            os.makedirs(custom_download_dir, exist_ok=True)

            pmid = row.get('pmid')

            if not pd.notna(row.get('is_oa')):
                pass
            elif pd.notna(row.get('is_oa')) or row.get('is_oa', False):
                for url in [row.get(f'pdf_url_{i}') for i in range(1, 5) if row.get(f'pdf_url_{i}')]:
                    if self.download_article_oa(url, custom_download_dir, pmid):
                        if self._update_download_status(custom_download_dir, index):
                            break  # Exit loop if successful download
                
            # If download is not complete, try PyPaperBot
            if self.df.at[index, 'download_complete'] != 'Complete' and allow_pypaperbot and row.get('doi') and pmid:
                if self.download_article_pypaperbot(row['doi'], pmid, custom_download_dir):
                    self._update_download_status(custom_download_dir, index)
                
            # If still not marked as complete, set as unavailable
            if self.df.at[index, 'download_complete'] != 'Complete':
                self.df.at[index, 'download_complete'] = "Unavailable"
                self.df.at[index, 'pdf_filepath'] = None
            
            if save_progress:
                self.save()

    def fetch_references(self):
        """
        Fetches references for each article in the DataFrame using the find_references method.

        The find_references method will attempt to fetch references in the following order:
        1. PubMed
        2. PMC
        3. Europe PMC
        4. CrossRef
        If no references are found using these methods, it will return "Not found".
        """
        if not hasattr(self, 'df') or self.df.empty:
            print("DataFrame does not exist or is empty.")
            return
        
        if 'references' not in self.df.columns:
            self.df['references'] = pd.NA
        
        for index, row in tqdm(self.df.iterrows(), total=self.df.shape[0], desc="Fetching References"):
            if pd.isna(row['references']):
                references = self._find_references_for_row(row)
                if not references:  # If references list is empty or None
                    references = "Not found"
                self.df.at[index, 'references'] = references

    def standardize_references(self):
        """
        Standardizes the references column in the DataFrame to only contain the following keys:
        ['doi', 'pmid', 'pmcid', 'title', 'authors']
        Populates a new column 'references_standardized' with the standardized references (list of dicts)
        """
        def standardize_references_for_row(references):
            standard_keys = ['doi', 'pmid', 'pmcid', 'title', 'authors']
            return [
                {key: ref.get(key, None) for key in standard_keys}
                for ref in references
                if isinstance(ref, dict)
            ]

        if 'references' not in self.df.columns:
            print('Error: No references column found in DataFrame.')
            return

        if 'references_standardized' not in self.df.columns:
            self.df['references_standardized'] = pd.NA

        for index, row in tqdm(self.df.iterrows(), total=self.df.shape[0], desc="Standardizing references"):
            ref_standardized = row['references_standardized']
            if pd.notna(ref_standardized) and (isinstance(ref_standardized, (list, np.ndarray)) and len(ref_standardized) > 0):
                continue
            
            references = row['references']
            if isinstance(references, list) and not references:
                continue
            if isinstance(references, np.ndarray) and references.size == 0:
                continue
            if not isinstance(references, (list, np.ndarray)) and pd.isna(references):
                continue

            self.df.at[index, 'references_standardized'] = standardize_references_for_row(references)

    def fetch_cited_by(self):
        """
        Fetches list of articles that cite each article in the DataFrame using Europe PMC.
        Currently only works for articles with a record in Europe PMC.
        """
        if not hasattr(self, 'df') or self.df.empty:
            print("DataFrame does not exist or is empty.")
            return
        
        if 'cited_by' not in self.df.columns:
            self.df['cited_by'] = pd.NA
        
        for index, row in tqdm(self.df.iterrows(), total=self.df.shape[0], desc="Fetching Cited By"):
            if pd.notna(row['cited_by']):
                continue
            if pd.notna(row.get('is_oa')) and row.get('is_oa') and pd.notna(row.get('europe_pmc_url')):
                cited_by = self.get_citing_articles_europe(row.get('pmid'))
                self.df.at[index, 'cited_by'] = cited_by

    def download_xml_fulltext(self, download_directory="downloads"):
        """
        Downloads the XML full text for each article in the DataFrame to the specified directory.

        Parameters:
        download_directory (str): The directory where the XML files should be saved.

        XML full text download is not very common (it has to be in the pubmed OA subset, either USA or European)
        """
        if not hasattr(self, 'df') or self.df.empty:
            print("DataFrame does not exist or is empty.")
            return
        
        if 'xml_download_complete' not in self.df.columns:
            self.df['xml_download_complete'] = 'Not started'
        if 'xml_filepath' not in self.df.columns:
            self.df['xml_filepath'] = None

        for index, row in tqdm(self.df.iterrows(), total=self.df.shape[0], desc="Downloading XML full texts"):
            if row.get('xml_download_complete') == 'Complete':
                continue
            
            custom_download_dir = self._determine_download_directory(row, download_directory, index)
            os.makedirs(custom_download_dir, exist_ok=True)

            last_name = row.get('first_author', '').split(',')[0].strip() if 'first_author' in row else None
            year = str(row.get('publication_year')) if 'publication_year' in row else None
            filename_suffix = f"{last_name}_{year}.xml" if last_name and year else None

            # Attempt to download XML full text
            if row.get('is_oa', False):
                file_path = None
                if pd.notna(row.get('europe_pmc_url')):
                    file_path = self.download_article_xml_europe(row.get('pmid'), custom_download_dir, filename_suffix)
                elif pd.notna(row.get('pmcid')):
                    file_path = self.download_article_xml_pubmed_oa_subset(row.get('pmcid'), custom_download_dir, filename_suffix)
                
                if file_path:
                    self.df.at[index, 'xml_download_complete'] = 'Complete'
                    self.df.at[index, 'xml_filepath'] = file_path
                else:
                    self.df.at[index, 'xml_download_complete'] = "Unavailable"
                    self.df.at[index, 'xml_filepath'] = None
            else:
                self.df.at[index, 'xml_download_complete'] = "Not OA or no XML available"
                self.df.at[index, 'xml_filepath'] = None

    def check_open_access(self):
        """
        Checks if each article in the DataFrame is open access using Unpaywall and fills in the relevant columns.
        Columns: 'is_oa', 'best_oa_location_url', 'pdf_url_1', 'pdf_url_2', 'pdf_url_3', 'pdf_url_4', 'europe_pmc_url'
        """
        required_columns = ['is_oa', 'best_oa_location_url', 'pdf_url_1', 'pdf_url_2', 
                            'pdf_url_3', 'pdf_url_4', 'europe_pmc_url']
        for column in required_columns:
            if column not in self.df.columns:
                self.df[column] = pd.NA

        # Check if 'doi' column exists
        if 'doi' not in self.df.columns:
            return "Error: 'doi' column not found in the DataFrame."

        # Iterate over DataFrame rows and fill in the values
        for index, row in tqdm(self.df.iterrows(), total=self.df.shape[0], desc="Checking Open Access"):
            if row.get('is_oa') is True or row.get('is_oa') is False:
                continue
            doi = row.get('doi')
            if pd.notna(doi):
                oa_info = self.check_open_access_doi(doi)
                for key in required_columns:
                    self.df.at[index, key] = oa_info.get(key, pd.NA)
            else:
                self.df.at[index, 'is_oa'] = False

    def save(self, csv_path="master_list.csv"):
        """
        Saves the DataFrame to a CSV file.
        """
        self.df.to_csv(csv_path, index=False)

    def save_abstracts_as_csv(self, filename="abstracts.csv"):
        """Saves a DataFrame containing only the 'pmid' and 'abstract' columns to a CSV file."""
        abstracts_df = self.df[['pmid','abstract']].copy()
        abstracts_df.to_csv(filename, index=False)

    def check_open_access_doi(self, doi):
        """
        Checks if an article is open access using Unpaywall and returns access information, including multiple PDF URLs.

        Parameters:
        doi (str): The DOI of the article to check.

        Returns:
        dict: A dictionary containing open access information, including if it's open access,
                the best link to access it, multiple links to download the PDF if available, and the OA status.
        """
        url = f"https://api.unpaywall.org/v2/{doi}?email={self.email}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            oa_status = data.get("oa_status", "unknown")
            is_oa = data.get("is_oa", False)
            best_oa_location_url = data.get("best_oa_location", {}).get("url", None) if data.get("best_oa_location") else None
            pdf_urls = [None, None, None, None]
            pdf_locations = [loc.get("url_for_pdf") for loc in data.get("oa_locations", []) if loc.get("url_for_pdf")]
            pdf_urls[:len(pdf_locations)] = pdf_locations[:4]

            pubmed_europe_info = next((
                (loc.get("url").split("?")[0], loc.get("url").split("pmc")[-1].split("/")[0])
                for loc in data.get("oa_locations", [])
                if "europepmc.org/articles/pmc" in loc.get("url", "")
            ), (None, None))

            pubmed_europe_url, pmcid = pubmed_europe_info

            return {
                "is_oa": is_oa,
                "best_oa_location_url": best_oa_location_url,
                "pdf_url_1": pdf_urls[0],
                "pdf_url_2": pdf_urls[1],
                "pdf_url_3": pdf_urls[2],
                "pdf_url_4": pdf_urls[3],
                "europe_pmc_url": pubmed_europe_url,
            }
        else:
            return {"error": f"Unpaywall API request failed with status code {response.status_code}"}

    def download_article_pypaperbot(self, doi, pmid, download_directory="downloads", mirror_idx=0):
        """
        Attempts to fetch the article using PyPaperBot based on the given DOI and Sci-Hub mirror.

        Parameters:
        doi (str): The DOI of the article to fetch.
        download_directory (str): The directory where the article PDF should be saved.
        mirror (str): The Sci-Hub mirror URL to use for downloading the article.

        Returns:
        str: A message indicating the result of the fetch operation.
        """
        mirror_list= ["https://sci-hub.st", "https://sci-hub.ru",  "https://sci-hub.se", "https://sci-hub.do"]   
        
        if mirror_idx >= len(mirror_list):
            return None
        
        try:
            command = f'PyPaperBot --doi {doi} --dwn-dir "{download_directory}" --scihub-mirror={mirror_list[mirror_idx]}'
            
            result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if self._check_if_downloaded(download_directory):
                self._rename_downloaded_file(download_directory, pmid)
                return "Article fetched successfully."
            else:
                return self.download_article_pypaperbot(doi, pmid, download_directory, mirror_idx + 1)
                               
        except subprocess.CalledProcessError as e:
            return f"Error executing PyPaperBot: {e}"
        
    def download_article_oa(self, pdf_url, download_directory, pmid=None):
        """
        Downloads an article from a provided open access PDF URL to the specified directory.
        Sets the filename as <last_name>_<year>.pdf if possible, or defaults to a generic name.

        Parameters:
        pdf_url (str): URL pointing to the PDF file.
        download_directory (str): The directory where the PDF file should be saved.
        last_name (str, optional): The last name of the first author. Defaults to None.
        year (str, optional): The publication year of the article. Defaults to None.
        """
        os.makedirs(download_directory, exist_ok=True)
        try:
            response = requests.get(pdf_url, stream=True)
            if response.status_code == 200:
                filename = f"{pmid}.pdf" if pmid else "downloaded_article.pdf"
                file_path = os.path.join(download_directory, filename)
                
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            else:
                return False
        except Exception:
            return False

    def get_references_europe(self, pmid):
        """
        Fetches references for an article identified by its PMID from Europe PMC API (hits the MEDLINE database).
        """
    
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/MED/{pmid}/references?page=1&pageSize=1000&format=json"

        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                data =  data.get('referenceList', [])
                if data:
                    parsed_data = data.get('reference', [])
                    if "id" in parsed_data:
                        parsed_data["pmid"] = parsed_data.pop("id")
                    if "authorString" in data:
                        parsed_data["authors"] = parsed_data.pop("authorString")
                return parsed_data.get('reference', []) if data else []
            else:
                # print(f"Failed to fetch references. Status code: {response.status_code}")
                return None
        except Exception as e:
            # print(f"Exception occurred while fetching references: {e}")
            return None
        
    def get_references_entrez_pmc(self, pmcid):
        """Finds references for a given PMCID using Entrez API.
        Seems to return identical results to the get_references_pubmed_oa_subset method.
        """
        Entrez.email = self.email
        handle = Entrez.efetch(db="pmc", id=pmcid, retmode="xml")
        xml_data = handle.read()
        handle.close()
        references = self._parse_pubmed_references(xml_data)
        return references
        
    def get_references_pubmed_oa_subset(self, pmcid):
        xml_content = self._get_xml_for_pmcid(pmcid)
        if xml_content:
            references = self._parse_pubmed_references(xml_content)
            return references
        else:
            return None
        
    def get_references_entrez_pubmed(self, pmid):
        """
        Returns a list of PMCIDs for the references of a given pmid. 
        Doesn't seem to work for all PMIDs, so use with caution.
        """
        Entrez.email = self.email
        handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
        article_details = Entrez.read(handle)
        handle.close()

        if article_details['PubmedArticle'][0]['PubmedData']['ReferenceList'] and len(article_details['PubmedArticle'][0]['PubmedData']['ReferenceList']) > 0:
            references = []
            try:
                # Attempt to navigate to the ReferenceList
                result_list = article_details['PubmedArticle'][0]['PubmedData']['ReferenceList'][0]['Reference']
                authors_pattern = r"^(.*?)\s+et al\."
                doi_pattern = r"doi\s*:\s*([^\s.]+)\.?"
                doi_pattern2 = r"doi\.org/([^\s,;]+)"
                doi_pattern3 = r"doi\.wiley\.org/([^\s,;]+)"

                for ref in result_list:
                    article_id_list = ref.get('ArticleIdList', [])
                    citation = ref.get('Citation', '')
                    ref_dict = {'citation': citation}

                    if article_id_list:
                        for element in article_id_list:
                            value = str(element)
                            id_type = element.attributes['IdType']
                            ref_dict[id_type] = value

                    if 'doi' not in ref_dict:
                        match = re.search(doi_pattern, citation, re.IGNORECASE)
                        if match:
                            ref_dict['doi'] = match.group(1)
                        elif 'doi' not in ref_dict:
                            match2 = re.search(doi_pattern2, citation, re.IGNORECASE)
                            if match2:
                                ref_dict['doi'] = match2.group(1)
                        else:
                            match3 = re.search(doi_pattern3, citation, re.IGNORECASE)
                            if match3:
                                ref_dict['doi'] = match3.group(1)

                    authors_match = re.search(authors_pattern, citation, re.IGNORECASE)
                    if authors_match:
                        ref_dict['authors'] = authors_match.group(1)

                    if 'pubmed' in ref_dict:
                        ref_dict['pmid'] = ref_dict.pop('pubmed')
                    if 'pmc' in ref_dict:
                        ref_dict['pmcid'] = ref_dict.pop('pmc')
                    
                    references.append(ref_dict)
                return references

            except (KeyError, IndexError, TypeError) as e:
                print(f"Error navigating article details: {e}")
                return None
        return None
        
    def get_references_crossref(self, doi):
        """
        Fetches references for a given DOI using the CrossRef REST API and formats them into a pandas DataFrame.
        
        Parameters:
            doi (str): The DOI of the article for which to fetch references.
            
        Returns:
            DataFrame: A pandas DataFrame containing the references, with each column representing a common key.
        """
        base_url = "https://api.crossref.org/works/"
        full_url = f"{base_url}{doi}"
        
        try:
            response = requests.get(full_url)
            if response.status_code == 200:
                data = response.json()
                references = data['message'].get('reference', [])
                if references:
                    for reference in references:
                        if 'DOI' in reference:
                            reference['doi'] = reference.pop('DOI')
                        if 'author' in reference:
                            reference['authors'] = reference.pop('author')
                        if 'article-title' in reference:
                            reference['title'] = reference.pop('article-title')
                    return references
                else:
                    return None
            else:
                print(f"Failed to fetch references, HTTP status code: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {str(e)}")
            return None

    def get_citing_articles_europe(self, pmid):
        """
        Fetches references for an article identified by its PMID from Europe PMC.
        Tries two different search methods and returns the results from the first successful one.
        """

        def try_restful_search(pmid):
            url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/MED/{pmid}/citations?format=json"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("citationList", {}).get("citation"):
                        parsed_data = data['citationList']
                        for citation in parsed_data["citation"]:
                            if "id" in citation:
                                citation["pmid"] = citation.pop("id")
                            if "authorString" in citation:
                                citation["authors"] = citation.pop("authorString")
                        return parsed_data["citation"]
                else:
                    print(f"Failed to fetch references. Status code: {response.status_code}")
            except Exception as e:
                print(f"Exception occurred while fetching references: {e}")
            return None

        def try_query_search(pmid):
            url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=cites:{pmid}_MED&format=json"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("resultList", {}).get("result"):
                        return data
                else:
                    print(f"Failed to fetch references. Status code: {response.status_code}")
            except Exception as e:
                print(f"Exception occurred while fetching references: {e}")
            return None

        # Try the RESTful search first
        restful_result = try_restful_search(pmid)
        if restful_result:
            return restful_result
        
        # If the RESTful search fails to provide results, try the query search
        query_result = try_query_search(pmid)
        if query_result:
            return query_result
        
        # If both searches fail, return a message indicating no results were found
        return "No citing articles found."
    
    def download_article_xml_europe(self, pmid, download_directory="downloads", filename_suffix=None):
        """
        Downloads the XML data for an article identified by its PMID from Europe PMC and saves it to a specified directory.
        """
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/MED/{pmid}/fullTextXML"
        print(url)

        try:
            response = requests.get(url)
            if response.status_code == 200:
                xml_content = response.text
                os.makedirs(download_directory, exist_ok=True)
                # Use filename_suffix if provided, else default to PMID
                filename = f"{filename_suffix}.xml" if filename_suffix else f"{pmid}.xml"
                file_path = os.path.join(download_directory, filename)
                
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(xml_content)
                
                print(f"Article XML downloaded successfully to {file_path}.")
                return file_path
            else:
                print(f"Failed to fetch article XML. Status code: {response.status_code}")
                return None
        except Exception as e:
            print(f"Exception occurred while fetching article XML: {e}")
            return None

    def download_article_xml_pubmed_oa_subset(self, pmcid, download_directory="downloads", filename_suffix=None):
        """
        Downloads the XML data for an article identified by its PMCID from PubMed OA subset and saves it to a specified directory.
        """
        xml_content = self._get_xml_for_pmcid(pmcid)
        if xml_content:
            os.makedirs(download_directory, exist_ok=True)
            # Use filename_suffix if provided, else default to PMCID
            filename = f"{filename_suffix}.xml" if filename_suffix else f"{pmcid}.xml"
            file_path = os.path.join(download_directory, filename)
            with open(file_path, 'wb') as file:
                file.write(xml_content)
            print(f"Article XML downloaded successfully to {file_path}.")
            return file_path
        else:
            print(f"Failed to download article XML for PMCID {pmcid}.")
            return None

    def _validate_dataframe(self, df):
        """Validates an input DataFrame to ensure it contains the required columns.
        Required columns: 'title', 'doi' # We may need to adjust this in the future
        """
        required_columns = ['title', 'doi']  # Adjusted required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame is missing required columns: {', '.join(missing_columns)}")

    def _parse_records_to_df(self, records_xml_bytes):
        """ Parses the API result Entrez into a DataFrame."""
        records_io = BytesIO(records_xml_bytes)
        records = Entrez.read(records_io)
        records_data = []

        for record in records['PubmedArticle']:
            article_data = {}
            medline = record['MedlineCitation']
            article = medline['Article']
            article_data['title'] = article.get('ArticleTitle', '')
            authors_list = article.get('AuthorList', [])
            authors = [f"{a.get('LastName', '')}, {a.get('ForeName', '')}" for a in authors_list]
            article_data['authors'] = "; ".join(authors)
            article_data['first_author'] = authors[0].split(',')[0] if authors else ''
            article_data['abstract'] = " ".join(article.get('Abstract', {}).get('AbstractText', []))
            publication_date = article.get('ArticleDate', [])
            article_data['publication_date'] = publication_date[0] if publication_date else {}
            article_data['publication_year'] = publication_date[0]['Year'] if publication_date else None
            article_data['journal_info'] = article.get('Journal', {}).get('Title', '')
            article_id_dict = {article_id.attributes['IdType']: str(article_id) for article_id in record.get('PubmedData', {}).get('ArticleIdList', [])}
            doi = article_id_dict.get('doi', "")
            pmcid = article_id_dict.get('pmc', "")
            pmid = medline.get('PMID', "")
            article_data['doi'] = doi
            article_data['pmcid'] = pmcid
            article_data['pmid'] = pmid

            keywords = medline.get('KeywordList', [])
            article_data['keywords'] = "; ".join([kwd for sublist in keywords for kwd in sublist]) if keywords else ""
            
            publication_types = article.get('PublicationTypeList', [])
            article_data['article_type'] = "; ".join([ptype for ptype in publication_types])
            medline_journal_info = medline.get('MedlineJournalInfo', {})
            article_data['country'] = medline_journal_info.get('Country', "")
            article_data['language'] = "; ".join(article.get('Language', []))

            records_data.append(article_data)

        df =  pd.DataFrame(records_data)
        cols = df.columns.tolist()
        cols.insert(0, cols.pop(cols.index('pmid')))
        df = df[cols]
        return df
    
    def _determine_download_directory(self, row, base_directory, index, enumerate=False):
        """Determine the download directory for an article based on its metadata."""
        def is_value_meaningful(value):
            return value and str(value).strip() not in ['', 'None', 'nan', 'NaN']
        
        pmid = row.get('pmid') if 'pmid' in row and is_value_meaningful(row['pmid']) else None

        dir_parts = []
        
        if enumerate:
            dir_parts.append(f"{index}")
        if pmid:
            dir_parts.append(f"{pmid}")

        directory_name = "_".join(dir_parts)
        return os.path.join(base_directory, directory_name)
    
    def _update_download_status(self, download_directory, index):
        """Update the download status of an article based on the downloaded PDF file."""
        if self._check_if_downloaded(download_directory):
            pdf_files = glob(os.path.join(download_directory, '*.pdf'))
            if pdf_files:
                self.df.at[index, 'download_complete'] = 'Complete'
                self.df.at[index, 'pdf_filepath'] = pdf_files[0]
                return True
        return False

    def _check_if_downloaded(self, download_directory_or_path, filetype=".pdf"):
        """Check if a file with the given extension exists in the specified directory or path."""
        files_with_type = glob(os.path.join(download_directory_or_path, f'*{filetype}'))
        return len(files_with_type) > 0
    
    def _rename_downloaded_file(self, download_directory, pmid):
        """Rename the downloaded file to match the article's PMID."""
        pdf_files = glob(os.path.join(download_directory, '*.pdf'))
        if pdf_files:
            pdf_file = pdf_files[0]
            new_pdf_file = os.path.join(download_directory, f"{pmid}.pdf")
            os.rename(pdf_file, new_pdf_file)
    
    def _find_references_for_row(self, row):
        pmcid = row.get('pmcid', None)  # Correct usage for a pandas Series
        pmid = row.get('pmid', None)
        doi = row.get('doi', None)

        def try_pubmed(pmid):
            if pmid:
                references = self.get_references_entrez_pubmed(pmid)
                if references is not None and len(references) > 0:
                    return references

        def try_pmc(pmcid):
            if pmcid:
                references = self.get_references_entrez_pmc(pmcid)
                if references is not None and len(references) > 0:
                    return references
                else:
                    references = self.get_references_pubmed_oa_subset(pmcid)
                    if references is not None and len(references) > 0:
                        return references

        def try_europe(pmid):
            if pmid:
                references = self.get_references_europe(pmid)
                if references is not None and len(references) > 0:
                    return references

        def try_crossref(doi):
            if doi:
                references = self.get_references_crossref(doi)
                if references is not None and len(references) > 0:
                    return references

        # Try all methods until one works
        references = try_pubmed(pmid)
        if references:
            return references

        references = try_pmc(pmcid)
        if references:
            return references

        references = try_europe(pmid)
        if references:
            return references

        references = try_crossref(doi)
        if references:
            return references

        # Return None or an empty list if no references found
        return []
        
    def _stringify_children(self, node):
        """
        Filters and removes possible Nones in texts and tails
        ref: http://stackoverflow.com/questions/4624062/get-all-text-inside-a-tag-in-lxml
        """
        parts = (
            [node.text]
            + list(chain(*([c.text, c.tail] for c in node.getchildren())))
            + [node.tail]
        )
        return "".join(filter(None, parts))

    def _parse_article_meta(self, tree):
        """
        Parse PMID, PMC and DOI from given article tree
        """
        article_meta = tree.find(".//article-meta")
        if article_meta is not None:
            pmid_node = article_meta.find('article-id[@pub-id-type="pmid"]')
            pmc_node = article_meta.find('article-id[@pub-id-type="pmc"]')
            pub_id_node = article_meta.find('article-id[@pub-id-type="publisher-id"]')
            doi_node = article_meta.find('article-id[@pub-id-type="doi"]')
        else:
            pmid_node = None
            pmc_node = None
            pub_id_node = None
            doi_node = None

        pmid = pmid_node.text if pmid_node is not None else ""
        pmc = pmc_node.text if pmc_node is not None else ""
        pub_id = pub_id_node.text if pub_id_node is not None else ""
        doi = doi_node.text if doi_node is not None else ""

        dict_article_meta = {"pmid": pmid, "pmc": pmc, "doi": doi, "publisher_id": pub_id}

        return dict_article_meta

    def _remove_namespace(self, tree):
        """
        Strip namespace from parsed XML
        """
        for node in tree.iter():
            try:
                has_namespace = node.tag.startswith("{")
            except AttributeError:
                continue  # node.tag is not a string (node is a comment or similar)
            if has_namespace:
                node.tag = node.tag.split("}", 1)[1]

    def _get_xml_for_pmcid(self, pmcid):
        """
        Fetches XML content for a given PMCID from PubMed Central.
        """
        pmcid = pmcid.replace("PMC", "")
        base_url = "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"
        params = {
            "verb": "GetRecord",
            "identifier": "oai:pubmedcentral.nih.gov:" + pmcid,
            "metadataPrefix": "pmc"
        }
        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            if 'is not supported by the item or by the repository.' in response.content.decode():
                print("Error: This PMC is not open access through PubMed Central, or the ID is invalid.")
                return None
            return response.content
        else:
            print("Error:", response.status_code)
            return None
        
    def _parse_pubmed_references(self, xml_content):
        """
        Parse reference articles from XML content to list of dictionaries, 
        independent of namespace.
        
        Parameters
        ----------
        xml_content: bytes
            XML content as bytes.
        
        Returns
        -------
        DataFrame
            A DataFrame containing references made in the given XML.
        """
        
        tree = etree.fromstring(xml_content)
        self._remove_namespace(tree)
        dict_article_meta = self._parse_article_meta(tree)
        pmid = dict_article_meta["pmid"]
        pmc = dict_article_meta["pmc"]
        references = tree.xpath(".//ref-list/ref")
        dict_refs = []

        for reference in references:
            ref_id = reference.attrib.get("id")
            ref_type = reference.xpath(".//citation/@citation-type")
            journal_type = ref_type[0] if ref_type else ""

            # Extract names
            names = reference.xpath(".//person-group[@person-group-type='author']/name/surname/text()") + \
                    reference.xpath(".//person-group[@person-group-type='author']/name/given-names/text()")
            names = [" ".join(names[i:i+2]) for i in range(0, len(names), 2)]
            
            # Extract article title, source, year, DOI, PMID
            article_title = reference.xpath(".//article-title/text()")
            article_title = article_title[0].replace("\n", " ").strip() if article_title else ""
            
            journal = reference.xpath(".//source/text()")
            journal = journal[0] if journal else ""
            
            year = reference.xpath(".//year/text()")
            year = year[0] if year else ""
            
            doi_cited = reference.xpath(".//pub-id[@pub-id-type='doi']/text()")
            doi_cited = doi_cited[0] if doi_cited else ""
            
            pmid_cited = reference.xpath(".//pub-id[@pub-id-type='pmid']/text()")
            pmid_cited = pmid_cited[0] if pmid_cited else ""
            
            dict_ref = {
                "pmid": pmid,
                "pmc": pmc,
                "ref_id": ref_id,
                "pmid_cited": pmid_cited,
                "doi_cited": doi_cited,
                "title": article_title,
                "authors": "; ".join(names),
                "year": year,
                "journal": journal,
                "journal_type": journal_type,
            }
            
            dict_refs.append(dict_ref)
            
        return dict_refs