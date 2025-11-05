## Quick script to count the number of individual reports in the text files
## Only useful for estimating cost per report, which is more reliable than cost per patient.

count=0
# with open('/Users/rm026/Documents/msa/msa_rpdr/RM026_073025133700885048_BWH_Prg.txt', 'r', encoding='utf-8') as file:
with open('/Users/rm026/Documents/hbs_study_patient_notes/RM026_120324154255294233_MGH_Dis.txt', 'r', encoding='utf-8') as file:
    for line in file:
        if 'report_end' in line:
            count += 1

print(count)