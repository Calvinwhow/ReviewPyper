################################################################################
# MODIFY THE VARIABLES BELOW TO FIT YOUR USE CASE! # 
################################################################################
# Set the file(s) you want to analyze (usually from an RPDR request) and the output directory
notes_file_list=['F:/Code/schmahmann_rpdr_results/rm026_110525111627789405_Dis.txt',
                 'F:/Code/schmahmann_rpdr_results/rm026_110525111627789405_Prg.txt',]

# Set the MRN file given by the RPDR request, to ensure proper matching of notes to subjects
# Some subjects may have multiple MRNs, and this ensure that all notes for a subject are included.
mrn_file='F:/Code/schmahmann_rpdr_results/rm026_110525111627789405_Mrn.txt'

output_dir='F:/Code/outputs/schmahmann_gpt5_fixed/'

# Provide the path to your OpenAI API key
api_key_path = "F:/Code/openai-key.txt"

# Provide the api base for your azure enclave, and the version you want to use
api_base = "https://mgb-risc-wrkspce-prod-e2-8-cog.openai.azure.com/"
api_version = "2025-01-01-preview"

# Provide the deployment names for the models used. 
# These are the names of your deployments on the azure enclave, NOT the name of the underlying model they use.
# The preprocessing model should be a very cheap model, since its jobs are simple. gpt-3.5-turbo is a good choice.
inclusion_model_name='gpt-4.1-mini'

# The extraction model, meanwhile, is for extracting the info you want from the files, 
# and needs to be more sophisticated. gpt-4 is a good choice.

extraction_model_name='gpt-5.1'

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
extraction_question_sets = ['bars','ccas','cnrs']

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
import json

inclusion_questions_json = json.load(open('inclusion_questions.json'))
inclusion_questions = {}
for questionnaire in inclusion_question_sets:
    inclusion_questions.update(inclusion_questions_json[questionnaire])
    
extraction_questions_json = json.load(open('extraction_questions.json', encoding='utf-8'))
extraction_questions={}
for questionnaire in extraction_question_sets:
    extraction_questions.update(extraction_questions_json[questionnaire])

master_list_path = output_dir+"master_list.csv"
master_list_excel_path = output_dir+"master_list.xlsx"
json_file_path = output_dir+"json/_emr_labeled_sections.json"

from calvin_utils.gpt_sys_review.txt_utils import ClinicalNotesExtractor
extractor=ClinicalNotesExtractor(notes_file_list, mrn_file, output_dir)
note_df=extractor.run()
extractor.generate_master_list()
extractor.save_master_list()

from calvin_utils.gpt_sys_review.txt_utils import TextPreprocessor
# Initialize the TextPreprocessor class and preprocess the files
preprocessor = TextPreprocessor(input_dir=output_dir)
preprocessed_path = preprocessor.process_files()

article_type = 'emr'  # 'case', 'research', 'emr', or 'other'

from calvin_utils.gpt_sys_review.json_utils import SectionLabeler
## Initialize the SectionLabeler class and process the files
## TODO: update this to check that the labeled sections file has all the 
## subjects in it, not just that it exists.
if os.path.exists(output_dir+"json/_emr_labeled_sections.json"):
    print(f"Found existing labeled sections at {output_dir+'json/_emr_labeled_sections.json'}. Skipping section labeling step.")
elif segment_file:
    section_labeler = SectionLabeler(folder_path=preprocessed_path, 
                                    article_type="emr", 
                                    api_key_path=api_key_path,)
    section_labeler.process_files()
else:
    # alternative: just create a json with the full text under 'emr' key
    # almost nothing gets counted as "other" in SectionLabeler anyways. 
    # Saves tons of time, cost is basically the same. 
    # TODO: clean this up later
    print("Skipping section labeling step.")
    section_json={}
    for filename in os.listdir(preprocessed_path):
        if not filename.endswith('.txt'):
            continue
        with open(os.path.join(preprocessed_path, filename)) as file:
            processed=file.read().replace('|', ',') 
            section_json[filename.split('.')[0]]={'emr':" ".join(processed.split())}

    os.mkdir(os.path.join(output_dir,'json'))
    with open(json_file_path, 'w') as f:
        json.dump(section_json, f, indent=0)


## Ask inclusion/exclusion questions
from calvin_utils.gpt_sys_review.gpt_utils.openai_json_evaluator import OpenAIJsonEvaluator
evaluator = OpenAIJsonEvaluator(api_key_path=api_key_path,
                                json_file_path=json_file_path, 
                                keys_to_consider=["emr"], 
                                answer_format='inclusion',
                                question_type='inclusion', 
                                model_choice=inclusion_model_name,
                                # include_explanations=True, # TODO: currently always includes explanations for inclusion questions, and this has to be set to True here. 
                                question=inclusion_questions, 
                                test_mode=test_mode,
                                debug=True,
                                is_azure=True, 
                                deployment_id=inclusion_model_name, 
                                api_version=api_version, 
                                api_base=api_base)
exclusion_answers = evaluator.evaluate_all_files()
new_json_path = evaluator.save_to_json(exclusion_answers)

from calvin_utils.gpt_sys_review.json_utils import InclusionExclusionSummarizer
summarizer = InclusionExclusionSummarizer(new_json_path, questions=inclusion_questions)
result_df, exclusion_raw_path = summarizer.run()


from calvin_utils.gpt_sys_review.txt_utils import PostProcessing
PostProcessing.update_emr_master_list(master_list_path=master_list_path, 
                                              raw_results_path=exclusion_raw_path,)
PostProcessing.rename_master_list_columns(master_list_path=master_list_path,
                                          questions_dict=inclusion_questions
                                          )


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
                                question_type="emr_strict_extraction",
                                question=extraction_questions,
                                answer_format=extraction_answer_format,
                                retain_chunks=True, 
                                # include_explanations=True,
                                test_mode=test_mode,
                                model_choice=extraction_model_name,
                                debug=extraction_debug,
                                max_workers=50,
                                is_azure=True,
                                deployment_id=extraction_model_name,
                                api_base=api_base,
                                api_version=api_version)
answers = evaluator.evaluate_all_files()
extraction_chunks_dir=evaluator.chunk_dir
evaluated_json_path = evaluator.save_to_json(answers)

# Perform Temporal Analysis
from calvin_utils.gpt_sys_review.gpt_utils.temporal_analysis import TemporalPlotter
plotter = TemporalPlotter(json_path=evaluated_json_path, output_dir=output_dir+"plots/")
onset_summary_path = plotter.run()

if onset_summary_path:
    from calvin_utils.gpt_sys_review.txt_utils import PostProcessing
    PostProcessing.update_emr_master_list(master_list_path=master_list_path, 
                                                  raw_results_path=onset_summary_path,)


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
                                     is_azure=True, 
                                     deployment_id=extraction_model_name, 
                                     api_base=api_base, api_version=api_version,
                                     debug=False,
                                    #  severity_mapping=severity_dict if extraction_answers_binary else None
                                     severity_mapping=severity_dict
                                    )
df, raw_path = custom_summarizer.run_custom(positive_explanations_only=True,)

from calvin_utils.gpt_sys_review.txt_utils import PostProcessing
PostProcessing.update_emr_master_list(master_list_path=master_list_path, 
                                              raw_results_path=raw_path,)
PostProcessing.rename_master_list_columns(master_list_path=master_list_path,
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

