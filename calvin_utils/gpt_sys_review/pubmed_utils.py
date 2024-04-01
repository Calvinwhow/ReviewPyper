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
    - fetch_cited_by: Fetches list of articles that cite each article in the DataFrame using Europe PMC (only works for articles with a record in Europe PMC)
    - download_xml_fulltext: Downloads the XML full text for each article in the DataFrame to the specified directory (rarely available, but can be useful).
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
        min_date (int): The minimum publication year to consider.
        max_date (int): The maximum publication year to consider.
        order_by (str): The order in which to retrieve articles. Can be 'chronological' or 'relevance'.
        """
        if not self.search_string:
            raise ValueError("Search string is not provided")

        Entrez.email = self.email
        search_handle = Entrez.esearch(db="pubmed", term=self.search_string, retmax=count,
                                        sort='relevance' if order_by == 'relevance' else 'pub date',
                                        mindate=str(min_date), maxdate=str(max_date))
        search_results = Entrez.read(search_handle)
        search_handle.close()

        id_list = search_results['IdList']
        fetch_handle = Entrez.efetch(db="pubmed", id=id_list, retmode="xml")
        records_xml_bytes = fetch_handle.read()
        fetch_handle.close()

        records_df = self._parse_records_to_df(records_xml_bytes)
        self.df = pd.concat([self.df, records_df], ignore_index=True)

    def download_articles(self, download_directory="downloads", allow_pypaperbot=True):
        """
        Downloads articles from the DataFrame to the specified directory. Tries to download open access articles first, then uses PyPaperBot as a fallback.

        Parameters:
        download_directory (str): The directory where the PDF files should be saved.
        allow_pypaperbot (bool): Whether to use PyPaperBot as a fallback for downloading articles.
        """

        if self.df.empty:
            print("DataFrame is empty.")
            return

        if 'download_complete' not in self.df.columns:
            self.df['download_complete'] = 'Not started'
        if 'pdf_filepath' not in self.df.columns:
            self.df['pdf_filepath'] = None

        for index, row in tqdm(self.df.iterrows(), total=self.df.shape[0], desc="Downloading articles"):
            if row.get('download_complete') == 'Complete':
                continue
            
            custom_download_dir = self._determine_download_directory(row, download_directory, index)
            os.makedirs(custom_download_dir, exist_ok=True)

            # Extract last name and year for filename
            last_name = row.get('first_author', '').split(',')[0].strip() if 'first_author' in row else None
            year = str(row.get('publication_year')) if 'publication_year' in row else None

            if row.get('is_oa', False):
                for url in [row.get(f'pdf_url_{i}') for i in range(1, 5) if row.get(f'pdf_url_{i}')]:
                    if self.download_article_oa(url, custom_download_dir, last_name, year):
                        if self._update_download_status(custom_download_dir, index):
                            break  # Exit loop if successful download
                
            # If download is not complete, try PyPaperBot
            if self.df.at[index, 'download_complete'] != 'Complete' and allow_pypaperbot and row.get('doi'):
                if self.download_article_pypaperbot(row['doi'], custom_download_dir):
                    self._update_download_status(custom_download_dir, index)
                
            # If still not marked as complete, set as unavailable
            if self.df.at[index, 'download_complete'] != 'Complete':
                self.df.at[index, 'download_complete'] = "Unavailable"
                self.df.at[index, 'pdf_filepath'] = None

    def fetch_references(self):
        """
        Fetches references for each article in the DataFrame using multiple methods.

        The references are fetched in the following order:
        1. Europe PMC (if available)
        2. PubMed OA Subset (if available)
        3. CrossRef (if DOI is available)
        4. "Not found" if no references are found using the above methods.
        """
        if not hasattr(self, 'df') or self.df.empty:
            print("DataFrame does not exist or is empty.")
            return
        
        if 'references' not in self.df.columns:
            self.df['references'] = pd.NA
        
        for index, row in tqdm(self.df.iterrows(), total=self.df.shape[0], desc="Fetching References"):
            references = None
            
            if pd.isna(row['references']):
                if row.get('is_oa') and pd.notna(row.get('europe_pmc_url')):
                    references = self.get_references_europe(row.get('pmid'))
                    
                if references is None or not references:
                    if row.get('is_oa') and pd.notna(row.get('pmcid')):
                        references = self.get_references_pubmed_oa_subset(row.get('pmcid'))
                
                if references is None or not references:
                    references = self.get_references_crossref(row.get('doi')) if pd.notna(row.get('doi')) else None
                
                if references is None or not references:
                    references = "Not found"
                
                self.df.at[index, 'references'] = references

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
            cited_by = None
            
            if pd.isna(row['cited_by']):
                if row.get('is_oa') and pd.notna(row.get('europe_pmc_url')):
                    cited_by = self.fetch_citing_articles_europe(row.get('pmid'))
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

    def check_open_access(self, doi):
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

    def download_article_pypaperbot(self, doi, download_directory="downloads", mirror="https://sci-hub.st"):
        """
        Attempts to fetch the article using PyPaperBot based on the given DOI and Sci-Hub mirror.

        Parameters:
        doi (str): The DOI of the article to fetch.
        download_directory (str): The directory where the article PDF should be saved.
        mirror (str): The Sci-Hub mirror URL to use for downloading the article.

        Returns:
        str: A message indicating the result of the fetch operation.
        """
        try:
            command = f'PyPaperBot --doi {doi} --dwn-dir "{download_directory}" --scihub-mirror={mirror}'
            
            result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            if result.returncode == 0:
                return "Article fetched successfully."
            else:
                return f"PyPaperBot encountered an error: {result.stderr}"
        except subprocess.CalledProcessError as e:
            return f"Error executing PyPaperBot: {e}"
        
    def download_article_oa(self, pdf_url, download_directory, last_name=None, year=None):
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
                filename = f"{last_name}_{year}.pdf" if last_name and year else "downloaded_article.pdf"
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
        Fetches references for an article identified by its PMID from Europe PMC.
        """
    
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/MED/{pmid}/references?format=json"

        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                return data
            else:
                print(f"Failed to fetch references. Status code: {response.status_code}")
                return None
        except Exception as e:
            print(f"Exception occurred while fetching references: {e}")
            return None
        
    def get_references_pubmed_oa_subset(self, pmcid):
        xml_content = self._get_xml_for_pmcid(pmcid)
        if xml_content:
            references = self._parse_pubmed_references(xml_content)
            return references
        else:
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
                    return references
                else:
                    print("No references found in the metadata.")
                    return None
            else:
                print(f"Failed to fetch references, HTTP status code: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {str(e)}")
            return None

    def fetch_citing_articles_europe(self, pmid):
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
                        return data
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

            if doi:
                oa_info = self.check_open_access(doi)
                article_data['is_oa'] = oa_info.get('is_oa', False)
                article_data['best_oa_location_url'] = oa_info.get('best_oa_location_url', '')
                article_data['pdf_url_1'] = oa_info.get('pdf_url_1', '')
                article_data['pdf_url_2'] = oa_info.get('pdf_url_2', '')
                article_data['pdf_url_3'] = oa_info.get('pdf_url_3', '')
                article_data['pdf_url_4'] = oa_info.get('pdf_url_4', '')
                article_data['europe_pmc_url'] = oa_info.get('europe_pmc_url', '')
            else:
                article_data['is_oa'] = False
                article_data['best_oa_location_url'] = ''
                article_data['best_oa_location_url_for_pdf'] = ''
                article_data['oa_status'] = 'unknown'

            keywords = medline.get('KeywordList', [])
            article_data['keywords'] = "; ".join([kwd for sublist in keywords for kwd in sublist]) if keywords else ""
            
            publication_types = article.get('PublicationTypeList', [])
            article_data['article_type'] = "; ".join([ptype for ptype in publication_types])
            medline_journal_info = medline.get('MedlineJournalInfo', {})
            article_data['country'] = medline_journal_info.get('Country', "")
            article_data['language'] = "; ".join(article.get('Language', []))

            records_data.append(article_data)

        return pd.DataFrame(records_data)
    
    def _determine_download_directory(self, row, base_directory, index):
        """Determine the download directory for an article based on its metadata."""
        def is_value_meaningful(value):
            return value and str(value).strip() not in ['', 'None', 'nan', 'NaN']

        first_author = row.get('first_author', '').split(',')[0].strip() if 'first_author' in row and is_value_meaningful(row['first_author']) else None
        publication_year = row.get('publication_year') if 'publication_year' in row and is_value_meaningful(row['publication_year']) else None
        pmid = row.get('pmid') if 'pmid' in row and is_value_meaningful(row['pmid']) else None

        dir_parts = [str(index)]

        if first_author:
            dir_parts.append(first_author)
        if publication_year:
            dir_parts.append(publication_year)
        elif pmid:
            dir_parts.append(f"pmid{pmid}")

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
                "article_title": article_title,
                "authors": "; ".join(names),
                "year": year,
                "journal": journal,
                "journal_type": journal_type,
            }
            
            dict_refs.append(dict_ref)
            
        return dict_refs

