import json

class QuestionTemplate:
    """
    Class to manage and display templates for various types of questions.
    
    # Example usage
    question_template = QuestionTemplate()
    question_template.inclusion_exclusion_questions()
    question_template.data_extraction_questions()
    question_template.print_question_template("inclusion")
    question_template.print_question_template("exclusion")
    question_template.print_question_template("custom")
    """
    def __init__(self):
        self.get_question_templates(self)
    
    def inclusion_exclusion_questions(self):
        """Method to print inclusion and exclusion questions."""

        print("Here are example inclusion questions:")
        print(json.dumps(self.inclusion_questions, indent=4))
        print("Here are example exclusion questions:")
        print(json.dumps(self.exclusion_questions, indent=4))
        print("Here is a question template")
        print(json.dumps(self.template_questions, indent=4))
    
    def data_extraction_questions(self):
        """Method to print data extraction questions."""
        
        print("Here are example data extraction questions:")
        print(json.dumps(self.strict_extraction_questions, indent=4))
        print("Here are some more generalizable data extraction questions:")
        print(json.dumps(self.lenient_extraction_questions, indent=4))
        print("Here is a question template")
        print(json.dumps(self.template_questions, indent=4))
    
    def get_question_templates(self, question_type):
        """
        Prints out a template for questions based on the specified question type.
        
        Parameters:
        - question_type (str): Type of questions to print ('inclusion', 'exclusion', 'evaluation').
        
        Returns:
        None
        """
        self.inclusion_questions = {'Amnesia case report? (Y/N)': 'case_report',
                            'Published in English? (Y/N)': 'is_english'}
        self.exclusion_questions = {'Transient amnesia, reversible amnesia symptom, severe confabulation or drug use, toxicity, epilepsy-related confusion, psychological or psychiatric-related amnesia (functional amnesia)': 'other_cause',
                            'Did not examine/report both retrograde and anterograde memory domains': 'not_both_domains',
                            'Without descriptive/qualitative/quantitative data on amnesia severity/memory tests/questions/scenarios/details': 'not_enough_information',
                            'Had global cognitive impairment disproportionate to memory loss': 'disproportionate_impairment',
                            'Without measurable lesion-related brain MR/CT scans': 'no_scan',
                            'Had focal or widespread brain atrophy': 'neurodegenerative',
                            'Atypical cases with selective (e.g., semantic) memory loss or material/topographic-specific memory loss': 'atypical_case'
                        }  
        self.strict_extraction_questions = {'Does the patient(s) represent(s) the whole experience of the investigator (center) or is the selection method unclear to the extent that other patients with similar presentation may not have been reported? (Good/Bad/Unclear)': 'representative_case_quality',
                            'Was patient’s causal exposure clearly described? (Good/Bad/Unclear)': 'causality_quality',
                            'Were diagnostic tests or assessment methods and the results clearly described (amnesia tests)? (Good/Bad/Unclear)': 'phenotyping_quality',
                            'Were other alternative causes that may explain the observation (amnesia) ruled out? (Good/Bad/Unclear)': 'workup_quality',
                            'Were patient’s demographics, medical history, comobidities clearly described? (Good/Bad/Unclear)': 'clinical_covariates_quality',
                            'Were patient’s symptoms, interventions, and clinical outcomes clearly presented as a timeline? (Good/Bad/Unclear)': 'history_quality',
                            'Was the lesion image taken around the time of observation (amnesia) assessment? (Good/Bad/Unclear)': 'temporal_causality_quality',
                            'Is the case(s) described with sufficient details to allow other investigators to replicate the research or to allow practitioners make inferences related to their own practice? (Good/Bad/Unclear)': 'history_quality_2'
            }
        self.lenient_extraction_questions = {'How affected is this study by selection bias? Little (Good). Very (Bad). Unsure (Unclear).': 'selection_bias',
                            'How well was the timeline of lesion/symptom onset described? (Good/Bad/Unclear)': 'exposure',
                            'How affected is this case by attribution error? Little (Good). Very (Bad). Unsure (Unclear).': 'outcome',
                            'How reasonable is the attribution of the lesion/diagnosis to the symptom? (Good/Bad/Unclear)': 'attribution_error',
                            'How well was the patients baseline pre-lesion described? (Good/Bad/Unclear)': 'pre_event_assessment',
                            'How well was the patients outcome post-lesion described? (Good/Bad/Unclear)': 'post_event_assessment',
                            'Do you think the neuroimaging was taken within temporal proximity to the lesion? Days-weeks (Good). Months-years (Bad). Unsure (Unclear).': 'lesion_related-images',
                            'Do you think another practitioner would come to the same conclusion (diagnosis/attribution) that this group did? (Good/Bad/Unclear)': 'replicability',
                            'How is the quality of this case overall? (Good/Bad/Unclear)': 'overall_appraisal'
            }        
        self.template_questions = {' ? (metric/metric/metric)': 'question_label',
                            ' ? (metric/metric/metric)': 'question_label',
                            ' ? (metric/metric/metric)': 'question_label',
                            ' ? (metric/metric/metric)': 'question_label',
                            ' ? (metric/metric/metric)': 'question_label',
                            ' ? (metric/metric/metric)': 'question_label',
                            ' ? (metric/metric/metric)': 'question_label',
                            ' ? (metric/metric/metric)': 'question_label',
            }
