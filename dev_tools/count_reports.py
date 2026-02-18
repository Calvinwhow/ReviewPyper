## Quick script to count the number of individual reports in the text files
## Only useful for estimating cost per report, which is more reliable than cost per patient.

count=0
# with open('/Users/rm026/Documents/msa/msa_rpdr/RM026_073025133700885048_BWH_Prg.txt', 'r', encoding='utf-8') as file:
with open('/Users/rm026/Documents/hbs_study_patient_notes/RM026_120324154255294233_MGH_Prg.txt', 'r', encoding='utf-8') as file:
    for line in file:
        if 'report_end' in line:
            count += 1
import json 
chunk_dict=json.load(open('/Users/rm026/Documents/hbs_study_patient_notes/hbs_seizure_reviewpyper_2-2-2026/json_evaluated/emr_strict_extraction_evaluations.json'))
chunkct=0
for mrn, mrn_chunks in chunk_dict.items():
    chunkct+=len(mrn_chunks['Has this patient had a seizure? For example, the text may mention epilepsy, convulsions, or fits. If none of these are mentioned in the text, respond "not present" instead of "no".'])

print('number of notes in prg:',count)
print('number of chunks from both prg and dis:', chunkct)