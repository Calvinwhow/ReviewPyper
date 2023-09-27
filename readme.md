![ReviewPyper Logo](assets/logo.png)


# Project Repository README

## Notebooks Overview

This repository contains a set of notebooks designed to assist with the processing and analysis of academic articles. Below is an overview of each notebook's purpose:

1. **Notebook 0: CSV File Processing**
   - This notebook receives a CSV file from a PubMed search and filters titles to extract relevant information.

2. **Notebook 1: Abstract Filtering**
   - Building upon the successful titles from the CSV, this notebook filters abstracts to narrow down the selection of papers.

3. **Notebook 2: PDF Review**
   - This notebook handles a directory of PDFs for a more in-depth review and processes them as needed.

4. **Notebook 3: Inclusion/Exclusion Analysis**
   - Performs an analysis to determine which papers meet the inclusion/exclusion criteria.

5. **Notebook 4: Data Extraction**
   - If acceptable keywords are mapped, this notebook will return a binarized spreadsheet of results. If not, it will provide raw results in a spreadsheet format. Additional manual data conversion may be required in the latter case.

## Important Warnings

- **Cost of GPT**: Please be aware that using GPT-based models for processing large volumes of data can incur significant costs. Be mindful of your budget when utilizing these notebooks.

- **Question Formatting**: Properly formatting questions is crucial for effective use of these notebooks. Ensure that your queries are clear and well-structured to obtain meaningful results.

- **Test Mode**: Before submitting an entire batch of articles, consider setting `test_mode=True` to iterate and refine your questions. This helps prevent submitting a batch of articles with poorly formulated queries.

Feel free to explore and use these notebooks for your academic article processing needs. If you have any questions or encounter issues, refer to the documentation or reach out for assistance.
