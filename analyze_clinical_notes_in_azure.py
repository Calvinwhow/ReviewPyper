################################################################################
# MODIFY THE VARIABLES BELOW TO FIT YOUR USE CASE! # 
################################################################################
# Set the file(s) you want to analyze (usually from an RPDR request) and the output directory
# notes_file_list=['F:/Code/schmahmann_rpdr_results/rm026_110525111627789405_Dis.txt',
#                  'F:/Code/schmahmann_rpdr_results/rm026_110525111627789405_Prg.txt',]
notes_file_list=['F:/code/hbs_fixed/hbs_combined_prg_dis_only_subs_w_imaging.txt']
# Set the MRN file given by the RPDR request, to ensure proper matching of notes to subjects
# Some subjects may have multiple MRNs, and this ensure that all notes for a subject are included.
# mrn_file='F:/Code/schmahmann_rpdr_results/rm026_110525111627789405_Mrn.txt'
mrn_file='F:/code/hbs_fixed/fixed_hbs_Mrn.txt'

#Optional: filter the MRNs used
import pandas as pd
select_mrns=pd.read_csv('F:/hbs_study_patient_notes/hbs_ross_aryan_gpt_n1205.csv')['MRN']
# select_mrns=None

output_dir='/Users/rm026/Documents/hbs_study_patient_notes/hbs_depression_memory_3/'

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

extraction_model_name='gpt-4.1'

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

skip_segmentation=False

# Set the questions for data extraction. This is where you extract what you want to know from the included notes.
# These are more open-ended than inclusion/exclusion questions, and don't have to be yes/no.
# See notebook 05, section 02 for examples.
# extraction_question_sets = ['bars','ccas','cnrs']
extraction_question_sets = ['hemiparesis','nih_stroke','depression','memory','moca','hbs_seizure']

# Types of answers you want for the extraction step. possible types are:
# - "binary_without_explanations"
# - "binary_with_explanations"
# - "binary_with_unknown_and_explanations"
# - "severity_with_explanations"
extraction_answer_format="binary_with_unknown_and_explanations"


# ################################################################################
# # DO NOT CHANGE ANYTHING BELOW THIS LINE UNLESS YOU KNOW WHAT YOU ARE DOING! #
# ################################################################################
import os
import json

############# Setup environment and set paths for output files #############class EmrEnvironment:
if os.path.exists(output_dir) is False:
    os.makedirs(output_dir)
    
from calvin_utils.gpt_sys_review.EmrEnvironment import EmrEnvironment
env=EmrEnvironment(output_dir+'env.json')
env.update({"notes_file_list": notes_file_list,
         "mrn_file": mrn_file,
         "output_dir": output_dir,
         "select_mrns": select_mrns,
         "api_key_path": api_key_path,
         "api_base": api_base,
         "api_version": api_version,
         "inclusion_model_name": inclusion_model_name,
         "extraction_model_name": extraction_model_name,
         "inclusion_question_sets": inclusion_question_sets,
         "extraction_question_sets": extraction_question_sets,
         "extraction_answer_format": extraction_answer_format,
         "test_mode": test_mode,
         "skip_segmentation": skip_segmentation,
         'extraction_completed': env.previous_run.get('extraction_completed', False),
         'section_labeling_step_completed': env.previous_run.get('section_labeling_step_completed', False),
         'inclusion_questions_completed': env.previous_run.get('inclusion_questions_completed', False),
         'inclusion_summarization_completed': env.previous_run.get('inclusion_summarization_completed', False),
         'extraction_questions_completed': False,
         'extraction_summarization_completed': False,
         'temporal_analysis_completed': False,
         'master_list_finalized': False
         })

# inclusion_questions_json = json.load(open('inclusion_questions.json', encoding='UTF-8'))
# inclusion_questions={q:name for set_name in inclusion_question_sets for q, name in inclusion_questions_json[set_name].items()}
    
extraction_questions_json = json.load(open('extraction_questions.json', encoding='UTF-8'))
extraction_questions={q:name for set_name in extraction_question_sets for q, name in extraction_questions_json[set_name].items()}
env.update({"extraction_questions": extraction_questions})

master_list_path = env.append_filenum(output_dir+"master_list.csv") 
master_list_excel_path = env.append_filenum(output_dir+"master_list.xlsx")
json_file_path = output_dir+"json/_emr_labeled_sections.json"
env.update({"master_list_path": master_list_path,
            "master_list_excel_path": master_list_excel_path,
            "json_file_path": json_file_path})

############# Preprocess text and split into individual notes #############

from calvin_utils.gpt_sys_review.txt_utils import PerNoteExtractor, TextPreprocessor
extractor=PerNoteExtractor(notes_file_list, mrn_file, output_dir)
preprocessor = TextPreprocessor(input_dir=output_dir)

if env.parameters['extraction_completed']==True:    
    note_df=extractor.run()
    preprocessed_path = preprocessor.process_files()
    env.update({'preprocessed_path': preprocessed_path})

else: # if extraction has already been done, just generate a blank master list 
    extractor.generate_master_list()
    extractor.save_master_list(run_counter=env.counter)
    preprocessed_path = preprocessor.output_dir
    env.update({'extraction_completed': True, 'preprocessed_path': preprocessed_path})

############# Exclude irrelevant text with SectionLabeler (currently skipping, by setting skip_segmentation=True) #############

from calvin_utils.gpt_sys_review.json_utils import SectionLabeler
## Initialize the SectionLabeler class and process the files
section_labeler = SectionLabeler(folder_path=preprocessed_path, 
                                article_type="emr", 
                                api_key_path=api_key_path,)
section_labeler.process_files(skip_segmentation=skip_segmentation)
env.update({'section_labeling_step_completed': True})

############# Ask inclusion/exclusion questions (currently skipped) #############

# # Ask inclusion/exclusion questions
# from calvin_utils.gpt_sys_review.gpt_utils.openai_json_evaluator import OpenAIJsonEvaluator
# evaluator = OpenAIJsonEvaluator(api_key_path=api_key_path,
#                                 json_file_path=json_file_path, 
#                                 keys_to_consider=["emr"], 
#                                 answer_format='inclusion',
#                                 question_type='inclusion', 
#                                 model_choice=inclusion_model_name,
#                                 # include_explanations=True, # TODO: currently always includes explanations for inclusion questions, and this has to be set to True here. 
#                                 question=inclusion_questions, 
#                                 test_mode=test_mode,
#                                 debug=True,
#                                 is_azure=True, 
#                                 deployment_id=inclusion_model_name, 
#                                 api_version=api_version, 
#                                 api_base=api_base)
# exclusion_answers = evaluator.evaluate_all_files()
# exclusion_anwers_json = evaluator.save_to_json(exclusion_answers)
# env.update({'exclusion_answers_json': exclusion_anwers_json, 'exclusion_questions_completed': True})

# from calvin_utils.gpt_sys_review.json_utils import InclusionExclusionSummarizer
# summarizer = InclusionExclusionSummarizer(exclusion_anwers_json, questions=inclusion_questions)
# result_df, exclusion_exclusion_summarizer_results_path = summarizer.run()
# from calvin_utils.gpt_sys_review.txt_utils import PostProcessing
# PostProcessing.update_emr_master_list(master_list_path=master_list_path, raw_results_path=exclusion_exclusion_summarizer_results_path,)
# env.update({'exclusion_exclusion_summarizer_results_path': exclusion_exclusion_summarizer_results_path, 'exclusion_summarization_completed': True})


############# Ask extraction questions #############

extraction_debug=True
if extraction_debug and os.path.exists('/Users/rm026/Documents/code/ReviewPyper/debug_openai_chat.txt'):
    os.remove('/Users/rm026/Documents/code/ReviewPyper/debug_openai_chat.txt')
elif extraction_debug and os.path.exists('/Users/rm026/Documents/code/ReviewPyper/error_log.txt'):
    os.remove('/Users/rm026/Documents/code/ReviewPyper/error_log.txt')
    
retain_chunks=True
max_workers=50
env.update({'retain_chunks': retain_chunks, 'max_workers': max_workers})

from calvin_utils.gpt_sys_review.gpt_utils.openai_json_evaluator import OpenAIJsonEvaluator
evaluator = OpenAIJsonEvaluator(api_key_path=api_key_path,
                                json_file_path=json_file_path, 
                                keys_to_consider=['emr'],
                                question_type="emr_strict_extraction",
                                question=extraction_questions,
                                answer_format=extraction_answer_format,
                                retain_chunks=retain_chunks, 
                                # include_explanations=True,
                                test_mode=test_mode,
                                model_choice=extraction_model_name,
                                max_workers=max_workers,
                                debug=extraction_debug,
                                is_azure=True,
                                deployment_id=extraction_model_name,
                                api_base=api_base,
                                api_version=api_version)
answers = evaluator.evaluate_all_files(chunk_by_date=True)
if answers is False:
    exit()
extraction_chunks_dir=evaluator.chunk_dir
evaluated_json_path = evaluator.save_to_json(answers)
env.update({'extraction_chunks_dir': extraction_chunks_dir, 'evaluated_json_path': evaluated_json_path, 'extraction_questions_completed': True})

############# Summarize extraction results #############

severity_dict = {
    0: ["unknown",'no info', 'not mentioned', 'no mention'],
    1: ["n", "no", "false",],
    2: ["y", "yes", "true", ]
}
summary_type='mapping'
env.update({'severity_dict': severity_dict, summary_type: summary_type})

from calvin_utils.gpt_sys_review.json_utils import CustomSummarizer
custom_summarizer = CustomSummarizer(json_path=evaluated_json_path, 
                                     answer_format=extraction_answer_format,
                                     summary_type=summary_type,
                                    #  chunks_dir=extraction_chunks_dir, 
                                     is_azure=True, 
                                     debug=False,
                                     severity_mapping=severity_dict,
                                     deployment_id=extraction_model_name, 
                                     api_base=api_base, api_version=api_version,
                                    )
df, exclusion_summarizer_results_path = custom_summarizer.run_custom(positive_explanations_only=True,)
env.update({'exclusion_summarizer_results_path': exclusion_summarizer_results_path, 'extraction_summarization_completed': True})

from calvin_utils.gpt_sys_review.txt_utils import PostProcessing
PostProcessing.update_emr_master_list(master_list_path=master_list_path, 
                                              raw_results_path=exclusion_summarizer_results_path,)

############# Perform Temporal Analysis #############

from calvin_utils.gpt_sys_review.gpt_utils.temporal_analysis import TemporalPlotter
plotter = TemporalPlotter(json_path=evaluated_json_path, output_dir=output_dir+"plots/")
onset_summary_path = plotter.run()
env.update({'onset_summary_path': onset_summary_path, 'temporal_analysis_completed': True})

from calvin_utils.gpt_sys_review.txt_utils import PostProcessing
PostProcessing.update_emr_master_list(master_list_path=master_list_path, 
                                      raw_results_path=onset_summary_path,)

############# Clean up outputs #############
PostProcessing.finalize_master_list(master_list_path=master_list_path, questions_dict=extraction_questions)
env.update({'master_list_finalized': True})

#Create excel file with masterListPath results
import pandas as pd
excelDf = pd.read_csv(master_list_path)
excelDf.to_excel(master_list_excel_path, index=False)

print("Done!")

