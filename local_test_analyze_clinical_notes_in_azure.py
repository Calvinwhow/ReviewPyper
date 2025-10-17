################################################################################
# MODIFY THE VARIABLES BELOW TO FIT YOUR USE CASE! # 
################################################################################

# Set the file you want to analyze (usually from an RPDR request) and the output directory
notes_file_list=['/Users/ahg26/Research/Collaborators/Calvin/calvinGithub/reviewPyper_working/data/00000016.txt']
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
# - A good paper will be denoted by 1, with a bad paper denoted by 0.
inclusion_questions = {
"Does this medical record include any information about a medical visit? For example, it might mention a movement or cognition task, a clinician's impression of the patient, or medical history information.": 1,
# "Prioritizing implicit and explicit information, does the patient have a documented seizure in their medical record?": 1,
# "Does this manuscript report memory outcomes? (Yes/No)": 0
}

# Set test_mode=True during your first few runs, while you tune your questions to get the answers you need
# - Always run this first, at least once. 
test_mode=False

# Set the questions for data extraction. This is where you extract what you want to know from the included notes.
# These are more open-ended than inclusion/exclusion questions, and don't have to be yes/no.
# See notebook 05, section 02 for examples.
# extraction_questions = {
#                         "Does this patient have difficulty walking or disturbed gait? For instance, they may mention the patient staggering, difficulties in half turn, requiring support from a wall or stick, or being entirely unable to walk on their own.":'gait',
#                         "Does this patient have difficulty performing the heel-shin maneuver such as lowering their heel jerkily, or with lateral movements?":'knee-tibia',
#                         "Does this patient’s speech show dysarthria or slurring to the point that some words are not intelligible? For instance, the text may mention having to ask the patient to repeat themselves multiple times. Do not consider hypophonia.":'dysarthria',
#                         "Does this patient have oculomotor abnormalities? For example, the text may mention slowed pursuit, saccadic intrusions, hypo/hypermetric saccade, or nystagmus.":'oculomotor',
#                         "Does this patient show dysmetria, oscillating movement, or segmented movement of the arm or hand when performing the finger-nose maneuver? Ignore any slowness.":'finger-nose',
#                         "Does this medical record state that the patient has had a seizure?":'seizure',
#                         "Is this patient afraid of having a seizure during the next month?":'seizure_fear',
#                         "Does this medical record state that the patient has had a stroke? Ignore any family histories of stroke or transient ischemic attacks. ":'stroke',
# }

extraction_questions = {
    # 0 = none/absent; 1 = mild; 2 = moderate; 3 = severe
    # IMPORTANT: Return only a single integer 0–3. No words, no punctuation.

    "For this patient, rate the severity of disturbed gait or walking difficulty (staggering, trouble turning, needs cane/wall, cannot walk unaided) on a 0–3 scale. 0=none/absent, 1=mild, 2=moderate, 3=severe. Return only a single integer between 0 and 3; if not mentioned, return 0.": "gait",
    "For this patient, rate the severity of difficulty performing heel–shin (jerky lowering or lateral movements) on a 0–3 scale. 0=none/absent, 1=mild, 2=moderate, 3=severe. Return only a single integer 0–3; if not mentioned, return 0.": "knee-tibia",
    "For this patient, rate the severity of dysarthria or slurred, partly unintelligible speech (not hypophonia) on a 0–3 scale. 0=none/absent, 1=mild, 2=moderate, 3=severe. Return only a single integer 0–3; if not mentioned, return 0.": "dysarthria",
    "For this patient, rate the severity of oculomotor abnormalities (slowed pursuit, saccadic intrusions, hypo/hypermetric saccades, nystagmus) on a 0–3 scale. 0=none/absent, 1=mild, 2=moderate, 3=severe. Return only a single integer 0–3; if not mentioned, return 0.": "oculomotor",
    "For this patient, rate the severity of finger–nose test abnormalities (dysmetria, oscillating/segmented movement; ignore slowness) on a 0–3 scale. 0=none/absent, 1=mild, 2=moderate, 3=severe. Return only a single integer 0–3; if not mentioned, return 0.": "finger-nose",
    "For this patient, rate the severity/presence of documented seizures on a 0–3 scale. 0=none/absent, 1=mild/past isolated mention, 2=moderate/recurrent or concerning, 3=severe/active or high clinical concern. Return only a single integer 0–3; if not mentioned, return 0.": "seizure",
    "For this patient, rate the severity/presence of fear of having a seizure in the next month on a 0–3 scale. 0=none/absent, 1=mild, 2=moderate, 3=severe. Return only a single integer 0–3; if not mentioned, return 0.": "seizure_fear",
    "For this patient, rate the severity/presence of documented stroke (exclude family history/TIA-only) on a 0–3 scale. 0=none/absent, 1=mild/past historical mention, 2=moderate/documented with some deficits, 3=severe/significant deficits. Return only a single integer 0–3; if not mentioned, return 0.": "stroke",
}

# - Set extraction_answers_binary to False if the extraction questions you asked do not have binary answers. 
#    - We will extract the raw data, like specific result values, for you to review.
# - Set extraction_answers_binary to True if the extraction questions you asked do have binary answers. 
#    - By default, we will set positive answers to 1, and negative answers to 0.
extraction_answers_binary=False
# ################################################################################
# # DO NOT CHANGE ANYTHING BELOW THIS LINE UNLESS YOU KNOW WHAT YOU ARE DOING! #
# ################################################################################
import os
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
                                model_choice="gpt3_small",
                                include_explanations=True,
                                question=inclusion_questions, 
                                test_mode=test_mode,
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
if extraction_debug and os.path.exists('/Users/rm026/Documents/code/ReviewPyper/debug_openai_chat.txt'):
    os.remove('/Users/rm026/Documents/code/ReviewPyper/debug_openai_chat.txt')
elif extraction_debug and os.path.exists('/Users/rm026/Documents/code/ReviewPyper/error_log.txt'):
    os.remove('/Users/rm026/Documents/code/ReviewPyper/error_log.txt')
    
from calvin_utils.gpt_sys_review.gpt_utils.openai_json_evaluator import OpenAIJsonEvaluator
evaluator = OpenAIJsonEvaluator(api_key_path=api_key_path,
                                json_file_path=json_file_path, 
                                keys_to_consider=['emr'],
                                question_type="emr_extraction",
                                question=extraction_questions,
                                retain_chunks=True, 
                                include_explanations=True,
                                test_mode=test_mode,
                                model_choice="gpt3_small",
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
custom_summarizer = CustomSummarizer(json_path=output_dir+"json_evaluated/emr_extraction_evaluations.json",
                                    #  json_path=evaluated_json_path, 
                                    # answers_binary=extraction_answers_binary, 
                                     answers_binary=False,
                                     summary_type='mapping', 
                                     api_key_path=api_key_path,
                                     chunks_dir=extraction_chunks_dir, 
                                     is_azure=False,
                                     severity_mapping=severity_dict)
df, raw_path = custom_summarizer.run_custom(positive_explanations_only=True,)

from calvin_utils.gpt_sys_review.txt_utils import PostProcessing
PostProcessing.add_raw_results_to_master_list(master_list_path=master_list_path, 
                                              raw_results_path=raw_path, 
                                              filename_col='MRN')

from calvin_utils.evaluate_iterations import IterationEvaluator
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

