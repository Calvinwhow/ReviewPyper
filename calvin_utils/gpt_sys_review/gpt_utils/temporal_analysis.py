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
        
        # If the JSON is a multi-patient dictionary of dictionaries
        for mrn, content in self.data.items():
            if not isinstance(content, dict): continue
            
            # The metadata might be in the patient block OR at the top level (global)
            metadata_by_chunk = content.get('metadata', self.data.get('metadata', {}))
            
            for question, chunks in content.items():
                # Skip system keys, explanations, and now Onset Dates
                if question in ['metadata', 'CHUNKS', 'MRN', 'filepath'] or question.startswith('EXPLANATION') or question.startswith('Onset Date:'):
                    continue
                
                if not isinstance(chunks, dict):
                    continue
                
                for chunk_id, answer in chunks.items():
                    ans = str(answer).lower()
                    if ans == '0' or ans == '0.0':
                        continue # Skip completely unmentioned symptom evaluations
                    
                    status = 1 # 1 = No
                    # More robust status check for Yes (2)
                    if any(s == ans for s in acceptable_strings) or ans == '2' or ans == '2.0':
                        status = 2
                    
                    # Extract date directly from the LLM's new inline output
                    # Fallback to chunk metadata if the LLM didn't return one or if running older files
                    date_str = str(content.get(f'Onset Date: {question}', {}).get(chunk_id, 'Unknown')).strip()
                    if date_str == 'Unknown' or date_str == '':
                        metadata = metadata_by_chunk.get(str(chunk_id), {})
                        date_str = metadata.get('date', 'Unknown')
                    
                    date_val = None
                    if date_str != 'Unknown' and date_str != 'None':
                        for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%Y/%m/%d'):
                            try:
                                date_val = datetime.strptime(date_str, fmt)
                                break

                            except ValueError:
                                continue
                        
                    records.append({
                        'MRN': mrn,
                        'Question': question,
                        'Chunk': chunk_id,
                        'RawDate': date_str,
                        'Date': date_val,
                        'Status': status
                    })
        
        df = pd.DataFrame(records)
        # Drop unknown dates for plotting - ensure it's not empty before returning
        if not df.empty:
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
        Generates a separate plot for EACH question for EACH participant.
        """
        if mrn:
            df = df[df['MRN'] == mrn]
        if questions:
            df = df[df['Question'].isin(questions)]
            
        unique_mrns = df['MRN'].unique()
        import textwrap
        
        for patient_id in unique_mrns:
            patient_df = df[df['MRN'] == patient_id]
            
            for i, question in enumerate(patient_df['Question'].unique()):
                plt.figure(figsize=(10, 5))
                q_df = patient_df[patient_df['Question'] == question].sort_values('Date')
                
                # Continuous step-plot for raw extraction
                plt.step(q_df['Date'], q_df['Status'], where='post', label="Raw Trajectory", color='black', linewidth=1.5, zorder=3)
                plt.scatter(q_df['Date'], q_df['Status'], alpha=0.8, s=40, color='red', label='Raw Extraction Point', zorder=4) 
                # Apply Option 2: Vertical Background Banding per Chunk (Contiguous Spans)
                # Apply Option 2: Vertical Background Banding per Chunk (Contiguous Spans) 
                unique_start_dates = sorted(q_df.groupby('Chunk')['Date'].min().unique())
                from datetime import timedelta
                
                # Use simple alternating distinct colors to denote different chunks cleanly
                # Light grey and light blue
                distinct_colors = ['#f5f5f5', '#e1f5fe'] * (len(unique_start_dates) + 1)
                
                if unique_start_dates:
                    # Determine borders for the bands
                    borders = unique_start_dates + [q_df['Date'].max() + timedelta(days=30)]
                    
                    for idx in range(len(borders) - 1):
                        span_start = borders[idx]
                        span_end = borders[idx+1]
                        
                        # Only plot if there's actual width
                        if span_start < span_end:
                            plt.axvspan(span_start, span_end, color=distinct_colors[idx], alpha=0.3, lw=0, zorder=1)
                
                title_text = textwrap.fill(question, width=80)
                plt.title(f"Patient {patient_id}\n{title_text}", fontsize=10)
                plt.xlabel("Date")
                plt.ylabel("Status (1=No, 2=Yes)")
                plt.ylim(0.8, 2.2)
                plt.yticks([1, 2], ['No (1)', 'Yes (2)'])
                
                # Move legend outside to prevent overlapping the plot content
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3, zorder=2)
                plt.tight_layout()
                
                safe_q = "".join([c if c.isalnum() else "_" for c in question[:30]]).strip("_")
                save_path = os.path.join(self.output_dir, f"temporal_{patient_id}_{safe_q}_{i}.png")
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()
                # print(f"Saved plot for {patient_id} - Q{i} to {save_path}")

    def summarize_onsets(self, df):
        """
        Extracts the first date where a symptom flipped to 'Yes' (Status 2).
        """
        # Filter for only 'Yes' statuses
        yes_df = df[df['Status'] == 2]
        
        # Get the first date for each MRN and Question
        onsets = yes_df.groupby(['MRN', 'Question'])['Date'].min().reset_index()
        onsets.rename(columns={'Date': 'OnsetDate'}, inplace=True)
        
        # Pivot to wider format: MRN, Q1_onset, Q2_onset...
        onsets_pivot = onsets.pivot(index='MRN', columns='Question', values='OnsetDate')
        
        # Clean up column names for readability in CSV
        onsets_pivot.columns = [f"Onset Date: {col}" for col in onsets_pivot.columns]
        return onsets_pivot

    def export_per_date_longitudinal_data(self, output_path):
        """
        Explodes the answer for each chunk across all valid dates found within that chunk.
        """
        records = []
        for mrn, content in self.data.items():
            if not isinstance(content, dict): continue
            metadata_by_chunk = content.get('metadata', self.data.get('metadata', {}))
            
            for question, chunks in content.items():
                if question in ['metadata', 'CHUNKS', 'MRN', 'filepath'] or question.startswith('EXPLANATION') or question.startswith('Onset Date:'):
                    continue
                if not isinstance(chunks, dict): continue
                
                for chunk_id, answer in chunks.items():
                    ans = str(answer).lower()
                    if ans == '0' or ans == '0.0':
                        continue # Exclude unmentioned data from the longitudinal tracking matrix
                        
                    status = 1 # 1 = No
                    if any(s == ans for s in ["2", "yes", "true", "present", "y"]) or ans == '2' or ans == '2.0':
                        status = 2 # 2 = Yes
                    
                    metadata = metadata_by_chunk.get(str(chunk_id), {})
                    all_dates = metadata.get('all_dates', [])
                    if not all_dates:
                        if metadata.get('date') and metadata.get('date') != "Unknown Date" and metadata.get('date') != "Unknown" and metadata.get('date') != "":
                            all_dates = [metadata.get('date')]
                        
                    for date_str in all_dates:
                        if isinstance(date_str, str) and date_str != 'Unknown Date' and date_str != 'Unknown':
                            records.append({
                                'MRN': mrn,
                                'Date': date_str,
                                'Question': question,
                                'Chunk': chunk_id,
                                'Status': status
                            })
                            
        if records:
            df = pd.DataFrame(records)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
            df.to_csv(output_path, index=False)
            return df
        return pd.DataFrame()

    def run(self):
        print("Extracting temporal data...")
        df = self.extract_temporal_data()
        
        long_path = os.path.join(self.output_dir, "longitudinal_per_date.csv")
        print(f"Exporting full per-date longitudinal tracking to {long_path}...")
        long_df = self.export_per_date_longitudinal_data(long_path)
        
        if long_df is None or long_df.empty:
            print("No valid temporal data (dates) found. Skipping plot/summary generation.")
            return None
            
        print(f"Applying permanent flip logic to {len(long_df)} exploded observations...")
        df_acc = self.apply_permanent_flip(long_df)
        
        # print("Generating plots...")
        # self.plot_patient_trajectories(df_acc)
        
        print("Summarizing onset dates...")
        summary_df = self.summarize_onsets(df)
        summary_path = os.path.join(self.output_dir, "temporal_onsets_summary.csv")
        summary_df.to_csv(summary_path)
        print(f"Saved onset summary to {summary_path}")
        
        return summary_path
