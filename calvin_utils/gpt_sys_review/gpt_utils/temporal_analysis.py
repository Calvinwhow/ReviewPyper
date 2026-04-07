import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class TemporalPlotter:
    """
    A class to analyze and plot the temporal progression of symptoms from evaluated JSON files.
    Implements a 'permanent flip' logic where once a symptom is detected, it stays detected.
    """

    def __init__(self, json_path, output_dir):
        self.json_path = json_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        with open(json_path, 'r') as f:
            self.data = json.load(f)

    def extract_temporal_data(self, acceptable_strings=["1", "yes", "true", "present", "y"]):
        """
        Parses the JSON data to extract dates and binary statuses for each question.
        Returns a dictionary or DataFrame with the results.
        """
        records = []
        for mrn, content in self.data.items():
            metadata_by_chunk = content.get('metadata', {})
            for question, chunks in content.items():
                if question == 'metadata' or question.startswith('EXPLANATION') or question.startswith('CHUNKS'):
                    continue
                
                for chunk_id, answer in chunks.items():
                    status = 0
                    ans = str(answer).lower()
                    metadata = metadata_by_chunk.get(chunk_id, {})
                    
                    if any(s in ans for s in acceptable_strings):
                        status = 1
                    
                    date_str = metadata.get('date', 'Unknown')
                    try:
                        date_val = datetime.strptime(date_str, '%m/%d/%Y')
                    except:
                        date_val = None
                        
                    records.append({
                        'MRN': mrn,
                        'Question': question,
                        'Chunk': chunk_id,
                        'RawDate': date_str,
                        'Date': date_val,
                        'Status': status
                    })
        
        df = pd.DataFrame(records)
        # Drop unknown dates for plotting
        df = df.dropna(subset=['Date'])
        return df

    def apply_permanent_flip(self, df):
        """
        Ensures that once a status becomes 1, it stays 1 for all future dates for that patient/question.
        """
        df = df.sort_values(by=['MRN', 'Question', 'Date'])
        df['AccumulatedStatus'] = df.groupby(['MRN', 'Question'])['Status'].cummax()
        return df

    def plot_patient_trajectories(self, df, mrn=None, questions=None):
        """
        Plots the trajectories for a specific patient or all patients.
        """
        if mrn:
            df = df[df['MRN'] == mrn]
        if questions:
            df = df[df['Question'].isin(questions)]
            
        unique_mrns = df['MRN'].unique()
        
        for patient_id in unique_mrns:
            patient_df = df[df['MRN'] == patient_id]
            plt.figure(figsize=(12, 6))
            
            for question in patient_df['Question'].unique():
                q_df = patient_df[patient_df['Question'] == question]
                plt.step(q_df['Date'], q_df['AccumulatedStatus'], where='post', label=question[:50] + "...")
                plt.scatter(q_df['Date'], q_df['Status'], alpha=0.3, s=10) # Show raw points faintly
            
            plt.title(f"Symptom Progression for Patient {patient_id} (Permanent Flip Logic)")
            plt.xlabel("Date")
            plt.ylabel("Status (0=No, 1=Yes)")
            plt.ylim(-0.1, 1.1)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            save_path = os.path.join(self.output_dir, f"temporal_{patient_id}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved plot for {patient_id} to {save_path}")

    def summarize_onsets(self, df):
        """
        Extracts the first date where a symptom flipped to 'Yes' (Status 1).
        """
        # Filter for only 'Yes' statuses
        yes_df = df[df['Status'] == 1]
        
        # Get the first date for each MRN and Question
        onsets = yes_df.groupby(['MRN', 'Question'])['Date'].min().reset_index()
        onsets.rename(columns={'Date': 'OnsetDate'}, inplace=True)
        
        # Pivot to wider format: MRN, Q1_onset, Q2_onset...
        onsets_pivot = onsets.pivot(index='MRN', columns='Question', values='OnsetDate')
        
        # Clean up column names for readability in CSV
        onsets_pivot.columns = [f"Onset Date: {col[:50]}..." for col in onsets_pivot.columns]
        return onsets_pivot

    def run(self):
        print("Extracting temporal data...")
        df = self.extract_temporal_data()
        if df.empty:
            print("No valid temporal data (dates) found. Skipping plot/summary generation.")
            return None
            
        print(f"Applying permanent flip logic to {len(df)} observations...")
        df_acc = self.apply_permanent_flip(df)
        
        print("Generating plots...")
        self.plot_patient_trajectories(df_acc)
        
        print("Summarizing onset dates...")
        summary_df = self.summarize_onsets(df)
        summary_path = os.path.join(self.output_dir, "temporal_onsets_summary.csv")
        summary_df.to_csv(summary_path)
        print(f"Saved onset summary to {summary_path}")
        
        return summary_path
