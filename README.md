# AITA Decision trees- predicting political leanings
Starting csv files: 
- Dataset Generation (Max) (Responses) - Form Responses 1.csv
- Dataset Generation Fall2025 (Responses) - Form Responses 1.csv

- Data cleaning notebooks are under the folder Datacleaning
- decisionTreeImages folder contains the decision trees
- ConfusionMatrixNormalize contains the confusion matrices used in the report

The following files were used to generate the decision trees, random forest and confusion matrices:
- DT_Fall_AITA.py
- DT_Fall_removed.py
- DT_Max_AITA.py
- DT_Max_removed.py

The files with 'removed' in their names are trained on the  cleaned datasets that exclude entries that indicate their political affiliation is 'don't know / It's complicated'

The extra folder contains decision trees where features also include non AITA questions.
