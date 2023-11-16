![ReviewPyper Logo](assets/logo.png)


# Project Repository README

**Installation**
   - Find the folder you saved this repository to. Copy it. 
   - Open your terminal and run the following command:
   - pip install -r /path/to/your/repository/requirements.txt

## Notebooks Overview

This repository contains a set of notebooks designed to assist with the processing and analysis of academic articles. Below is an overview of each notebook's purpose:

0. **Notebook 0: CSV File Processing**
   - This notebook receives a CSV file from a PubMed search and filters titles to extract relevant information.

1. **Notebook 1: Abstract Filtering**
   - Building upon the successful titles from the CSV, this notebook filters abstracts to narrow down the selection of papers.

2. **Notebook 2: PDF Review**
   - This notebook handles a directory of PDFs for a more in-depth review and processes them as needed.

3. **Notebook 4: Inclusion/Exclusion Analysis**
   - This notebook handles segmentation of manuscript sections. It allows you to create JSON which has all sections of a manuscript labelled. Currently, the manuscript types are 'case' or 'research'.
   - Case will automatically segment manuscripts into 'case_report' or 'other'.
   - 'research' will allow you to segment manuscripts into 'Abstract', 'Introduction', 'Methods', 'Results', 'Discussion', 'Conclusion'm and 'References'.

4. **Notebook 4: Inclusion/Exclusion Analysis**
   - Performs an analysis to determine which papers meet the inclusion/exclusion criteria.
   - You can choose which segments of manuscripts to provide for evaluation. For example, references are quite expensive and do not contain data relevant to a case report, and can often be excluded. 

5. **Notebook 5: Data Extraction**
   - If acceptable keywords are mapped, this notebook will return a binarized spreadsheet of results. If not, it will provide raw results in a spreadsheet format. Additional manual data conversion may be required in the latter case.

## Important Warnings

- **Cost of GPT**: Please be aware that using GPT-based models for processing large volumes of data can incur significant costs. Be mindful of your budget when utilizing these notebooks.

- **Question Formatting**: Properly formatting questions is crucial for effective use of these notebooks. Ensure that your queries are clear and well-structured to obtain meaningful results.

- **Test Mode**: Before submitting an entire batch of articles, consider setting `test_mode=True` to iterate and refine your questions. This helps prevent submitting a batch of articles with poorly formulated queries.

Feel free to explore and use these notebooks for your academic article processing needs. If you have any questions or encounter issues, refer to the documentation or reach out for assistance.
