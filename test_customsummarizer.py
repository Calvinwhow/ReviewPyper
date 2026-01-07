import unittest
import shutil
import os
import pandas as pd
from calvin_utils.gpt_sys_review.json_utils import CustomSummarizer

class TestCustomSummarizer(unittest.TestCase):
        
     def setUp(self):
          
          self.template_dir = '/Users/rm026/Documents/code/ReviewPyper/unit_testing/test_case/'
          self.test_dir = '/Users/rm026/Documents/code/ReviewPyper/unit_testing/temp/'
          if os.path.exists(self.test_dir):
               shutil.rmtree(self.test_dir)

          os.makedirs(self.test_dir + 'json_evaluated', exist_ok=True)

          shutil.copy(self.template_dir + 'json_evaluated/emr_strict_extraction_evaluations.json',
                    self.test_dir + 'json_evaluated/emr_strict_extraction_evaluations.json')
          
          shutil.copy(self.template_dir + 'blank_master_list.csv',
                    self.test_dir + 'master_list.csv')

          severity_dict = {
               0: ["unknown",'no info', 'no information'],
               1: ["n", "no", "false"],
               2: ["y", "yes", "true", "present"]
          }
          self.summarizer = CustomSummarizer(json_path=self.test_dir + 'json_evaluated/emr_strict_extraction_evaluations.json',
                                        answer_format="binary_with_unknown_and_explanations",
                                        summary_type='mapping', 
                                        api_key_path="",
                                        is_azure=False,
                                        severity_mapping=severity_dict
                                        )

     def test_run_custom(self):

          df, raw_path = self.summarizer.run_custom(positive_explanations_only=True)

          self.assertIsNotNone(df)
          self.assertIsNotNone(raw_path)
          self.assertTrue(len(df) > 0)

          responses_raw=pd.read_csv(raw_path)

          expected_response_columns = ['Unnamed: 0','Gait','EXPLANATION: Gait',
                                        'Heel-shin','EXPLANATION: Heel-shin',
                                        'Speech','EXPLANATION: Speech',
                                        'Oculomotor','EXPLANATION: Oculomotor',
                                        'Finger-nose','EXPLANATION: Finger-nose']
          self.assertEqual(responses_raw.columns.tolist(),expected_response_columns)

          self.assertTrue(type(responses_raw['EXPLANATION: Gait'][0])==str)
          self.assertNotEqual(responses_raw['EXPLANATION: Gait'][0],'')

          self.assertEqual(int(responses_raw['Gait'][0]), 2) # Yes, Yes
          self.assertEqual(int(responses_raw['Speech'][0]), 1) # No, No
          self.assertEqual(int(responses_raw['Heel-shin'][0]), 0) # Unknown, Unknown
          self.assertEqual(int(responses_raw['Oculomotor'][0]), 2) # Yes, Unknown
          self.assertEqual(int(responses_raw['Finger-nose'][0]), 2) # Yes, No

     


if __name__ == '__main__':
    unittest.main()

