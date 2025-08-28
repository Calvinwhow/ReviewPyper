################################################################################
# MODIFY THE VARIABLES BELOW TO FIT YOUR USE CASE! # 
################################################################################

# Set the file you want to analyze (usually from an RPDR request) and the output directory
notes_file_list=['/Users/rm026/Documents/hbs_study_patient_notes/test_patient_file_anonymized.txt',
                 ]
output_dir='/Users/rm026/Documents/hbs_study_patient_notes/py_testing_output_2/'

# Provide the path to your OpenAI API key
api_key_path = "/Users/rm026/Documents/code/openai-key.txt"


# Define inclusion/exclusion questions. See notebook 04, section 01 for examples.
# **Critical Note**
# - You are going to define a dictionary with questions as keys (first) and outcomes as values (second).
# - The value determines if you are answering a positive question or a negative question.
# - If the question is positive (a yes is good), set the value to 1.
# - If the question is negative (a yes is bad), set the value to 0.
# - A good paper will be denoted by 1, with a bad paper denoted by 0.
inclusion_questions = {
"Prioritizing implicit and explicit information, does this medical record mention a stroke? For example, the text may directly mention stroke, ischemia, or an infarct. (Yes/No)": 1,
# "Prioritizing implicit and explicit information, does the patient have a documented seizure in their medical record? (Yes/No)": 1,
# "Does this manuscript report memory outcomes? (Yes/No)": 0
}

# Set test_mode=True during your first few runs, while you tune your questions to get the answers you need
# - Always run this first, at least once. 
test_mode=False

# Set the questions for data extraction. This is where you extract what you want to know from the included notes.
# These are more open-ended than inclusion/exclusion questions, and don't have to be yes/no.
# See notebook 05, section 02 for examples.
extraction_questions = {
    "Prioritizing implicit and explicit information, report the date in MM/DD/YY format that the patient experienced their first stroke?": "stroke_date",
    "Prioritizing implicit and explicit information, does the patient have a documented seizure in their medical record? ": "seizure",
    }

# - Set extraction_answers_binary to False if the extraction questions you asked do not have binary answers. 
#    - We will extract the raw data, like specific result values, for you to review.
# - Set extraction_answers_binary to True if the extraction questions you asked do have binary answers. 
#    - By default, we will set positive answers to 1, and negative answers to 0.
extraction_answers_binary=False

################################################################################
# DO NOT CHANGE ANYTHING BELOW THIS LINE UNLESS YOU KNOW WHAT YOU ARE DOING! #
################################################################################

from calvin_utils.gpt_sys_review.txt_utils import ClinicalNotesExtractor
extractor=ClinicalNotesExtractor(notes_file_list, output_dir)
note_df=extractor.run()

from calvin_utils.gpt_sys_review.txt_utils import TextPreprocessor
# Initialize the TextPreprocessor class and preprocess the files
preprocessor = TextPreprocessor(input_dir=output_dir)
preprocessed_path = preprocessor.process_files()

article_type = 'emr'  # 'case', 'research', 'emr', or 'other'
master_list_path = output_dir+"master_list.csv"


from calvin_utils.gpt_sys_review.json_utils import SectionLabeler
# Initialize the SectionLabeler class and process the files
section_labeler = SectionLabeler(folder_path=preprocessed_path, article_type="emr", api_key_path=api_key_path)
section_labeler.process_files()


keys_to_consider = [ "emr"]  # Add or remove keys as per your requirement
article_type = 'inclusion'


json_file_path = output_dir+"json/_emr_labeled_sections.json"

from calvin_utils.gpt_sys_review.gpt_utils.openai_json_evaluator import OpenAIJsonEvaluator
evaluator = OpenAIJsonEvaluator(api_key_path=api_key_path, json_file_path=json_file_path, keys_to_consider=keys_to_consider, question_type=article_type, model_choice="gpt4",  question=inclusion_questions, test_mode=test_mode)
exclusion_answers = evaluator.evaluate_all_files()
new_json_path = evaluator.save_to_json(exclusion_answers)

from calvin_utils.gpt_sys_review.json_utils import InclusionExclusionSummarizer
summarizer = InclusionExclusionSummarizer(new_json_path, questions=inclusion_questions)
result_df, exclusion_raw_path = summarizer.run()


from calvin_utils.gpt_sys_review.txt_utils import PostProcessing
PostProcessing.add_raw_results_to_master_list(master_list_path=master_list_path, raw_results_path=exclusion_raw_path, filename_col='MRN')


csv_path = output_dir+"json_evaluated/inclusion_exclusion_results.csv"

from calvin_utils.gpt_sys_review.json_utils import FilterPapers

# Initialize and run the FilterPapers class
filter_papers = FilterPapers(csv_path=csv_path, json_path=json_file_path)
filtered_json_path = filter_papers.run()


from calvin_utils.gpt_sys_review.gpt_utils.openai_json_evaluator import OpenAIJsonEvaluator
evaluator = OpenAIJsonEvaluator(api_key_path=api_key_path,
                                json_file_path=json_file_path, 
                                keys_to_consider=keys_to_consider,
                                question_type="extraction",
                                question=extraction_questions,
                                test_mode=test_mode,
                                model_choice="gpt4",
                                debug=False)
answers = evaluator.evaluate_all_files()
evaluated_json_path = evaluator.save_to_json(answers)


from calvin_utils.gpt_sys_review.json_utils import CustomSummarizer
custom_summarizer = CustomSummarizer(json_path=evaluated_json_path, answers_binary=extraction_answers_binary, summary_type='llm', api_key_path=api_key_path)
df, raw_path = custom_summarizer.run_custom()

PostProcessing.add_raw_results_to_master_list(master_list_path=master_list_path, raw_results_path=raw_path, filename_col='MRN')
