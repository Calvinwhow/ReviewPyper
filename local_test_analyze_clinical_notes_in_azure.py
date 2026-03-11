################################################################################
# MODIFY THE VARIABLES BELOW TO FIT YOUR USE CASE! # 
################################################################################

# Set the file you want to analyze (usually from an RPDR request) and the output directory
notes_file_list=['/Users/ahg26/Research/Collaborators/Calvin/calvinGithub/reviewPyper_working/data/msa_all_sub_notes_prg_and_dis_deidentified.txt']
output_dir='/Users/ahg26/Research/Collaborators/Calvin/calvinGithub/reviewPyper_working/output/'

#Additional files for auc calculation
ground_truth_path = "/Users/ahg26/Research/Collaborators/Calvin/calvinGithub/reviewPyper_working/data/redacted_msa_ground_truth_no_maybes.csv"
iteration_history_path = "/Users/ahg26/Research/Collaborators/Calvin/calvinGithub/reviewPyper_working/output/all_patients_gpt3_iteration_history.csv"
accuracy_image = output_dir+"iteration_accuracy.png"

# Provide the path to your OpenAI API key
api_key_path = "/Users/ahg26/Research/Collaborators/Calvin/calvinGithub/reviewPyper_working/data/openai_key.txt"


# Define inclusion/exclusion questions. See notebook 04, section 01 for examples.
# **Critical Note**
# - You are going to define a dictionary with questions as keys (first) and outcomes as values (second).
# - The value determines if you are answering a positive question or a negative question.
# - If the question is positive (a yes is good), set the value to 1.
# - If the question is negative (a yes is bad), set the value to 0.
# - A good note will be denoted by 1, with a bad note denoted by 0.
inclusion_question_sets = ['emr_inclusion']

# Set test_mode=True during your first few runs, while you tune your questions to get the answers you need
# - Always run this first, at least once. 
test_mode=False

# Set the questions for data extraction. This is where you extract what you want to know from the included notes.
# These are more open-ended than inclusion/exclusion questions, and don't have to be yes/no.
# See notebook 05, section 02 for examples.
extraction_question_sets = ['bars', 'test_questions']

# - Set extraction_answers_binary to False if the extraction questions you asked do not have binary answers. 
#    - We will extract the raw data, like specific result values, for you to review.
# - Set extraction_answers_binary to True if the extraction questions you asked do have binary answers. 
#    - By default, we will set positive answers to 1, and negative answers to 0.
extraction_answers_binary=True

# ################################################################################
# # DO NOT CHANGE ANYTHING BELOW THIS LINE UNLESS YOU KNOW WHAT YOU ARE DOING! #
# ################################################################################
import os
import json

inclusion_questions_json = json.load(open('inclusion_questions.json'))
inclusion_questions = {}
for questionnaire in inclusion_question_sets:
    inclusion_questions.update(inclusion_questions_json[questionnaire])
    
extraction_questions_json = json.load(open('extraction_questions.json'))
extraction_questions={}
for questionnaire in extraction_question_sets:
    extraction_questions.update(extraction_questions_json[questionnaire])

master_list_path = output_dir+"master_list.csv"
json_file_path = output_dir+"json/_emr_labeled_sections.json"

from calvin_utils.gpt_sys_review.txt_utils import ClinicalNotesExtractor
extractor=ClinicalNotesExtractor(notes_file_list, output_dir)
note_df=extractor.run()

from calvin_utils.gpt_sys_review.txt_utils import TextPreprocessor
# Initialize the TextPreprocessor class and preprocess the files
preprocessor = TextPreprocessor(input_dir=output_dir)
preprocessed_path = preprocessor.process_files()

article_type = 'emr'  # 'case', 'research', 'emr', or 'other'


from calvin_utils.gpt_sys_review.json_utils import SectionLabeler
## Initialize the SectionLabeler class and process the files
## TODO: update this to check that the labeled sections file has all the subjects in it, not just that it exists.
if os.path.exists(output_dir+"json/_emr_labeled_sections.json"):
    print(f"Found existing labeled sections at {output_dir+'json/_emr_labeled_sections.json'}. Skipping section labeling step.")
else:
    section_labeler = SectionLabeler(folder_path=preprocessed_path, 
                                    article_type="emr", 
                                    api_key_path=api_key_path)
    section_labeler.process_files()

# Ask inclusion/exclusion questions
from calvin_utils.gpt_sys_review.gpt_utils.openai_json_evaluator import OpenAIJsonEvaluator
evaluator = OpenAIJsonEvaluator(api_key_path=api_key_path,
                                json_file_path=json_file_path, 
                                keys_to_consider=["emr"], 
                                question_type='inclusion', 
                                model_choice="gpt4.1",
                                include_explanations=True,
                                question=inclusion_questions, 
                                test_mode=test_mode,
                                response_tokens=2000,
                                debug=True)
exclusion_answers = evaluator.evaluate_all_files()
new_json_path = evaluator.save_to_json(exclusion_answers)

from calvin_utils.gpt_sys_review.json_utils import InclusionExclusionSummarizer
summarizer = InclusionExclusionSummarizer(new_json_path, questions=inclusion_questions)
result_df, exclusion_raw_path = summarizer.run()


from calvin_utils.gpt_sys_review.txt_utils import PostProcessing
PostProcessing.add_raw_results_to_master_list(master_list_path=master_list_path, 
                                              raw_results_path=exclusion_raw_path, 
                                              filename_col='MRN')


csv_path = output_dir+"json_evaluated/inclusion_exclusion_results.csv"

extraction_debug=True
if extraction_debug and os.path.exists('debug_openai_chat.txt'):
    os.remove('debug_openai_chat.txt')
elif extraction_debug and os.path.exists('error_log.txt'):
    os.remove('error_log.txt')
    
from calvin_utils.gpt_sys_review.gpt_utils.openai_json_evaluator import OpenAIJsonEvaluator
evaluator = OpenAIJsonEvaluator(api_key_path=api_key_path,
                                json_file_path=json_file_path, 
                                keys_to_consider=['emr'],
                                question_type="emr_extraction",
                                question=extraction_questions,
                                retain_chunks=True, 
                                include_explanations=True,
                                test_mode=test_mode,
                                response_tokens=8000,
                                model_choice="gpt4.1",
                                max_workers=50,
                                debug=extraction_debug)
answers = evaluator.evaluate_all_files()
extraction_chunks_dir=evaluator.chunk_dir
evaluated_json_path = evaluator.save_to_json(answers)


severity_dict = {
    0: ["none", "absent", "no"],
    1: ["mild", "slight"],
    2: ["moderate"],
    3: ["severe", "marked", "significant"]
}

from calvin_utils.gpt_sys_review.json_utils import CustomSummarizer
custom_summarizer = CustomSummarizer(json_path=evaluated_json_path,
                                     answers_binary=extraction_answers_binary, 
                                     summary_type='mapping', 
                                     api_key_path=api_key_path,
                                     chunks_dir=extraction_chunks_dir, 
                                     is_azure=False,
                                     max_workers=50,
                                     severity_mapping=severity_dict if not extraction_answers_binary else None)
df, raw_path = custom_summarizer.run_custom(positive_explanations_only=False,)

# Renaming Stroke Column in the raw results to match Ground Truth BEFORE merging into master list
gt_stroke_col = "EXPLANATION: Does this medical record state that the patient has had a stroke? Ignore any family histories of stroke or transient ischemic attacks. "
short_stroke_col = "Does this medical record state that the patient has had a stroke? Ignore any family histories of stroke or transient ischemic attacks. "

import pandas as pd
raw_df = pd.read_csv(raw_path, index_col=0)
if short_stroke_col in raw_df.columns:
    if gt_stroke_col in raw_df.columns:
        raw_df.drop(columns=[gt_stroke_col], inplace=True)
    raw_df.rename(columns={short_stroke_col: gt_stroke_col}, inplace=True)
    raw_df.to_csv(raw_path)
    print(f"Renamed {short_stroke_col} to {gt_stroke_col} in {raw_path} to match Ground Truth.")

# Ensure master list has 'MRN' column name (fixes potential header issues)
master_df_temp = pd.read_csv(master_list_path)
if 'MRN' not in master_df_temp.columns:
    master_df_temp.rename(columns={master_df_temp.columns[0]: 'MRN'}, inplace=True)
    master_df_temp.to_csv(master_list_path, index=False)

from calvin_utils.gpt_sys_review.txt_utils import PostProcessing
PostProcessing.add_raw_results_to_master_list(master_list_path=master_list_path, 
                                              raw_results_path=raw_path, 
                                              filename_col='MRN')

# Commented out due to column mismatch with new severity-scale questions
# from calvin_utils.evaluate_iterations import IterationEvaluator
from calvin_utils.evaluate_iterations import IterationEvaluator # Corrected import path
ev = IterationEvaluator(
    ground_path=ground_truth_path,
    master_path=master_list_path,
    history_path=iteration_history_path,
    plot_path=accuracy_image,
    reset=False,
)
row = ev.run()
print(row)

print("Done!")

