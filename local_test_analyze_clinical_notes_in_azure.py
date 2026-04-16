################################################################################
# MODIFY THE VARIABLES BELOW TO FIT YOUR USE CASE! # 
################################################################################
# Set the file(s) you want to analyze (usually from an RPDR request) and the output directory
notes_file_list=['/Users/rm026/Documents/Code/reviewpyper_testing/00000016.txt',]
# notes_file_list=["/Users/rm026/Documents/hbs_study_patient_notes/fixed_hbs/all_files/rm026_021126133356272397_Prg.txt",
#                  '/Users/rm026/Documents/hbs_study_patient_notes/fixed_hbs/all_files/rm026_021126133356272397_Dis.txt',
#                  '/Users/rm026/Documents/hbs_study_patient_notes/fixed_hbs/all_files/RM026_120324154255294233_MGH_Prg.txt',
#                  '/Users/rm026/Documents/hbs_study_patient_notes/fixed_hbs/all_files/RM026_120324154255294233_MGH_Dis.txt']
output_dir='/Users/rm026/Documents/Code/reviewpyper_testing/tests/fix_hbs_only_mrns_w_imgs/'

# mrn_file="/Users/rm026/Documents/Code/reviewpyper_testing/msa_fake_mrn_file.txt"
mrn_file='/Users/rm026/Documents/hbs_study_patient_notes/fixed_hbs/fixed_hbs_Mrn.txt'

#Optional: filter the MRNs used
import pandas as pd
df=pd.read_csv('/Users/rm026/Documents/hbs_study_patient_notes/sub_id_to_mrn_included.csv',dtype=str)
select_mrns=pd.concat([df['MRN_alt'],df['MRN_primary']]).unique().tolist()
select_mrns=[x for x in select_mrns if type(x)==str]
print(len(select_mrns))
# select_mrns=None

# Provide the path to your OpenAI API key
api_key_path = "/Users/rm026/Documents/code/openai-key.txt"

# Define inclusion/exclusion questions. See notebook 04, section 01 for examples.
# **Critical Note**
# - You are going to define a dictionary with questions as keys (first) and outcomes as values (second).
# - The value determines if you are answering a positive question or a negative question.
# - If the question is positive (a yes is good), set the value to 1.
# - If the question is negative (a yes is bad), set the value to 0.
# - A good note will be denoted by 1, with a bad note denoted by 0.
inclusion_question_sets =['emr_inclusion']

# Set test_mode=True during your first few runs, while you tune your questions to get the answers you need
# - Always run this first, at least once. 
test_mode=False

segment_file=False

# Set the questions for data extraction. This is where you extract what you want to know from the included notes.
# These are more open-ended than inclusion/exclusion questions, and don't have to be yes/no.
# See notebook 05, section 02 for examples.
extraction_question_sets = ['bars','hemiparesis']

# Types of answers you want for the extraction step. possible types are:
# - "binary_without_explanations"
# - "binary_with_explanations"
# - "binary_with_unknown_and_explanations"
# - "severity_with_explanations"
extraction_answer_format="binary_with_unknown_and_explanations"

evaluate_accuracy=False
#Additional files for auc calculation
ground_truth_path = "/Users/rm026/Documents/code/ReviewPyper_testing/redacted_msa_ground_truth_no_maybes.csv"
iteration_history_path = "/Users/rm026/Documents/code/ReviewPyper_testing/all_patients_gpt4_bars_redacted_iteration_history.csv"
accuracy_image = output_dir+"iteration_accuracy.png"

# ################################################################################
# # DO NOT CHANGE ANYTHING BELOW THIS LINE UNLESS YOU KNOW WHAT YOU ARE DOING! #
# ################################################################################
import os
import sys
# Force use of local workspace files instead of installed site-packages
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
import json

inclusion_questions_json = json.load(open('inclusion_questions.json', encoding='UTF-8'))
inclusion_questions={q:name for set_name in inclusion_question_sets for q, name in inclusion_questions_json[set_name].items()}
    
extraction_questions_json = json.load(open('extraction_questions.json'))
extraction_questions={q:name for set_name in extraction_question_sets for q, name in extraction_questions_json[set_name].items()}

master_list_path = output_dir+"master_list.csv"
master_list_excel_path = output_dir+"master_list.xlsx"
json_file_path = output_dir+"json/_emr_labeled_sections.json"

from calvin_utils.gpt_sys_review.txt_utils import ClinicalNotesExtractor, TextPreprocessor
counter=0
while os.path.isfile(master_list_path): #rename if master list already exists
    counter+=1
    master_list_path='/'.join(master_list_path.split('/')[:-1]+[f'master_list_{counter}.csv'])

extractor=ClinicalNotesExtractor(notes_file_list, mrn_file, output_dir, filter_list=select_mrns, debug=True)
preprocessor = TextPreprocessor(input_dir=output_dir)
if counter==0:    
    note_df=extractor.run()
    preprocessed_path = preprocessor.process_files()
else: # if extraction has already been done, just generate a blank master list 
    extractor.generate_master_list()
    extractor.save_master_list()
    preprocessed_path = preprocessor.output_dir

from calvin_utils.gpt_sys_review.json_utils import SectionLabeler
## Initialize the SectionLabeler class and process the files
## TODO: update this to check that the labeled sections file has all the 
## subjects in it, not just that it exists.
if os.path.exists(output_dir+"json/_emr_labeled_sections.json"):
    print(f"Found existing labeled sections at {output_dir+'json/_emr_labeled_sections.json'}. Skipping section labeling step.")
else:
    section_labeler = SectionLabeler(folder_path=preprocessed_path, 
                                    article_type="emr", 
                                    api_key_path=api_key_path,
                                    label_files=False)
    section_labeler.process_files(label_files=segment_file)


# Ask inclusion/exclusion questions
# from calvin_utils.gpt_sys_review.gpt_utils.openai_json_evaluator import OpenAIJsonEvaluator
# evaluator = OpenAIJsonEvaluator(api_key_path=api_key_path,
#                                 json_file_path=json_file_path, 
#                                 keys_to_consider=["emr"], 
#                                 answer_format='inclusion',
#                                 question_type='inclusion', 
#                                 model_choice="gpt-4.1-mini", # TODO: Need to find a new cheap model for this that's not in 
#                                 # include_explanations=True, # TODO: currently always includes explanations for inclusion questions, and this has to be set to True here. 
#                                 question=inclusion_questions, 
#                                 test_mode=test_mode,
#                                 response_tokens=2000,
#                                 debug=False)
# exclusion_answers = evaluator.evaluate_all_files()
# new_json_path = evaluator.save_to_json(exclusion_answers)


# from calvin_utils.gpt_sys_review.json_utils import InclusionExclusionSummarizer
# summarizer = InclusionExclusionSummarizer(new_json_path, questions=inclusion_questions)
# result_df, exclusion_raw_path = summarizer.run()


# from calvin_utils.gpt_sys_review.txt_utils import PostProcessing
# PostProcessing.update_emr_master_list(master_list_path=master_list_path, 
#                                               raw_results_path=exclusion_raw_path)

extraction_debug=False
if extraction_debug and os.path.exists('debug_openai_chat.txt'):
    os.remove('debug_openai_chat.txt')
elif extraction_debug and os.path.exists('error_log.txt'):
    os.remove('error_log.txt')
    
from calvin_utils.gpt_sys_review.gpt_utils.openai_json_evaluator import OpenAIJsonEvaluator
evaluator = OpenAIJsonEvaluator(api_key_path=api_key_path,
                                json_file_path=json_file_path, 
                                keys_to_consider=['emr'],
                                question_type="emr_strict_extraction",
                                question=extraction_questions,
                                answer_format=extraction_answer_format,
                                retain_chunks=True, 
                                # include_explanations=True,
                                test_mode=test_mode,
                                # response_tokens=8000,
                                model_choice="gpt-5.1",
                                max_workers=50,
                                debug=extraction_debug)
answers = evaluator.evaluate_all_files()
extraction_chunks_dir=evaluator.chunk_dir
evaluated_json_path = evaluator.save_to_json(answers)



severity_dict = {
    0: ["unknown",'no info', 'no information', 'not mentioned', 'not present'],
    1: ["n", "no", "false"],
    2: ["y", "yes", "true", ]
}

from calvin_utils.gpt_sys_review.json_utils import CustomSummarizer
custom_summarizer = CustomSummarizer(json_path=output_dir+"json_evaluated/emr_strict_extraction_evaluations.json",
                                    #  json_path=evaluated_json_path, 
                                     answer_format=extraction_answer_format,
                                     summary_type='mapping', 
                                     api_key_path=api_key_path,
                                    #  chunks_dir=extraction_chunks_dir, 
                                     is_azure=False,
                                    #  debug=True,
                                    #  severity_mapping=severity_dict if extraction_answers_binary else None
                                     severity_mapping=severity_dict
                                    )
df, raw_path = custom_summarizer.run_custom(positive_explanations_only=True,)

from calvin_utils.gpt_sys_review.txt_utils import PostProcessing
PostProcessing.update_emr_master_list(master_list_path=master_list_path, 
                                              raw_results_path=raw_path,)

# Perform Temporal Analysis
from calvin_utils.gpt_sys_review.gpt_utils.temporal_analysis import TemporalPlotter
plotter = TemporalPlotter(json_path=evaluated_json_path, output_dir=output_dir+"/plots/")
onset_summary_path = plotter.run()

if onset_summary_path:
    from calvin_utils.gpt_sys_review.txt_utils import PostProcessing
    PostProcessing.update_emr_master_list(master_list_path=master_list_path, 
                                                  raw_results_path=onset_summary_path)


PostProcessing.finalize_master_list(master_list_path=master_list_path,
                                          questions_dict=extraction_questions
                                          )

#Create excel file with masterListPath results
import pandas as pd
excelDf = pd.read_csv(master_list_path)
excelDf.to_excel(master_list_excel_path, index=False)

if evaluate_accuracy:
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

