import argparse
import re
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from calvin_utils.gpt_sys_review.gpt_utils.temporal_analysis import TemporalPlotter

class SymptomProgressionPlotter:
    """
    Generate symptom progression step plots aligned to clinical onset date.

    Expected inputs:
        1. ReviewPyper JSON output.
        2. Onset CSV containing MRN and an onset/stroke date column.
        3. One or more exact ReviewPyper JSON symptom/question keys to plot.

    The class:
        - extracts longitudinal data from ReviewPyper JSON via TemporalPlotter
        - loads onset data from CSV
        - validates required columns
        - standardizes MRNs and dates
        - merges review data with onset dates
        - computes months from onset
        - plots one interactive trace per MRN on each symptom-key plot
    """
    TRACE_COLORS = {
        "YES BEFORE ONSET | YES AFTER ONSET": "#1f77b4",
        "YES BEFORE ONSET | NO AFTER ONSET": "#ff7f0e",
        "NO BEFORE ONSET | NO AFTER ONSET": "#7f7f7f",
        "NO BEFORE ONSET | YES AFTER ONSET": "#2ca02c",
    }
    TRACE_Y_OFFSETS = {
        "YES BEFORE ONSET | YES AFTER ONSET": 0.05,
        "YES BEFORE ONSET | NO AFTER ONSET": 0.0,
        "NO BEFORE ONSET | NO AFTER ONSET": -0.05,
        "NO BEFORE ONSET | YES AFTER ONSET": 0.0,
    }

    def __init__(
        self,
        json_path,
        onset_csv_path,
        symptoms,
        output_dir="dateTimePlots",
        onset_col="stroke_date",
        date_col="Date",
        mrn_col="MRN",
        temp_longitudinal_filename="temp_longitudinal.csv",
        min_valid_year=1900,
        max_valid_year=2100,
        drop_unknown=False,
        show_yes_before_yes_after=True,
        show_yes_before_no_after=True,
        show_no_before_no_after=True,
        show_no_before_yes_after=True,
        display_plot=True,
        transition_threshold_months=3,
    ):
        self.json_path = Path(json_path)
        self.onset_csv_path = Path(onset_csv_path)
        self.symptom_keys = list(symptoms)
        self.output_dir = Path(output_dir)
        self.requested_onset_col = onset_col
        self.onset_col = onset_col
        self.date_col = date_col
        self.mrn_col = mrn_col
        self.temp_longitudinal_path = self.output_dir / temp_longitudinal_filename
        self.min_valid_year = min_valid_year
        self.max_valid_year = max_valid_year
        self.drop_unknown = drop_unknown
        self.transition_threshold_months = transition_threshold_months
        self.enabled_trace_conditions = {
            "YES BEFORE ONSET | YES AFTER ONSET": show_yes_before_yes_after,
            "YES BEFORE ONSET | NO AFTER ONSET": show_yes_before_no_after,
            "NO BEFORE ONSET | NO AFTER ONSET": show_no_before_no_after,
            "NO BEFORE ONSET | YES AFTER ONSET": show_no_before_yes_after,
        }
        self.display_plot = display_plot

        self.review_df = None
        self.onset_df = None
        self.merged_df = None
        self.series_df = None

    def run(self):
        """
        Full orchestration method.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.load_review_data()
        self.load_onset_data()
        self.resolve_onset_column()
        self.prepare_data()
        fig, output_path = self.plot_all()
        if self.display_plot:
            self.display_figure(fig)

        print(f"Plot saved to {output_path}")
        return fig, output_path

    def load_review_data(self):
        """
        Extract longitudinal per-date symptom data from ReviewPyper JSON.
        """
        plotter = TemporalPlotter(
            json_path=str(self.json_path),
            output_dir=str(self.output_dir),
        )
        self.validate_json_symptom_keys(plotter.data)

        self.review_df = plotter.export_per_date_longitudinal_data(
            str(self.temp_longitudinal_path),
            symptom_keys=self.symptom_keys,
        )

        if self.review_df is None or self.review_df.empty:
            raise ValueError("No valid longitudinal data could be extracted from JSON.")

    def validate_json_symptom_keys(self, data):
        """
        Validate requested symptom keys against the evaluated JSON before doing
        the expensive per-date export.
        """
        available_keys = set()
        skipped_keys = {'metadata', 'CHUNKS', 'MRN', 'filepath'}

        for content in data.values():
            if not isinstance(content, dict):
                continue
            for key, value in content.items():
                if key in skipped_keys or key.startswith('EXPLANATION') or key.startswith('onset_date:'):
                    continue
                if isinstance(value, dict):
                    available_keys.add(key)

        missing = [key for key in self.symptom_keys if key not in available_keys]
        if not missing:
            return

        available = "\n".join(f"  - {key}" for key in sorted(available_keys))
        raise ValueError(
            "Requested symptom keys were not found in the evaluated JSON: "
            f"{missing}\nAvailable symptom keys:\n{available}"
        )

    def load_onset_data(self):
        """
        Load onset CSV.
        """
        self.onset_df = pd.read_csv(self.onset_csv_path)

        if self.onset_df.empty:
            raise ValueError("Onset CSV is empty.")

    def resolve_onset_column(self):
        """
        Resolve the onset date column.

        Priority:
            1. Exact requested column.
            2. First column containing 'stroke' and either 'date' or 'onset'.
            3. Column named 'symptom_onset'.
        """
        if self.requested_onset_col in self.onset_df.columns:
            self.onset_col = self.requested_onset_col
            return

        possible_cols = [
            col for col in self.onset_df.columns
            if "stroke" in col.lower()
            and ("date" in col.lower() or "onset" in col.lower())
        ]

        if possible_cols:
            self.onset_col = possible_cols[0]
            print(f"Auto-selected onset column: '{self.onset_col}'")
            return

        if "symptom_onset" in self.onset_df.columns:
            self.onset_col = "symptom_onset"
            print(f"Auto-selected onset column: '{self.onset_col}'")
            return

        available = "\n".join(f"  - {col}" for col in self.onset_df.columns)
        raise ValueError(
            f"Onset CSV must contain '{self.requested_onset_col}' or a recognizable "
            f"stroke/onset date column.\nAvailable columns:\n{available}"
        )

    def prepare_data(self):
        """
        Validate, standardize, merge, filter, and compute months since onset.
        """
        self.validate_required_columns()
        self.standardize_mrn_columns()
        self.coerce_date_columns()
        self.trim_onset_columns()
        self.merge_data()
        self.drop_missing_dates()
        self.filter_invalid_dates()
        self.compute_months_since_onset()
        self.build_symptom_series()

        if self.merged_df.empty:
            raise ValueError("No valid data remaining after merging and date filtering.")

    def validate_required_columns(self):
        """
        Validate required columns in review and onset dataframes.
        """
        self.require_columns(
            self.review_df,
            [self.mrn_col, self.date_col],
            dataframe_name="Extracted JSON longitudinal data",
        )

        self.require_columns(
            self.onset_df,
            [self.mrn_col, self.onset_col],
            dataframe_name="Onset CSV",
        )

    @staticmethod
    def require_columns(df, columns, dataframe_name):
        """
        Raise an explicit error if required columns are missing.
        """
        missing = [col for col in columns if col not in df.columns]

        if missing:
            available = "\n".join(f"  - {col}" for col in df.columns)
            raise ValueError(
                f"{dataframe_name} is missing required columns: {missing}\n"
                f"Available columns:\n{available}"
            )

    def standardize_mrn_columns(self):
        """
        Ensure MRN columns have matching string type and whitespace stripping.
        """
        self.review_df[self.mrn_col] = (
            self.review_df[self.mrn_col].astype(str).str.strip()
        )

        self.onset_df[self.mrn_col] = (
            self.onset_df[self.mrn_col].astype(str).str.strip()
        )

    def coerce_date_columns(self):
        """
        Convert review dates and onset dates to pandas datetime.
        """
        self.review_df[self.date_col] = pd.to_datetime(
            self.review_df[self.date_col],
            errors="coerce",
        )

        self.onset_df[self.onset_col] = pd.to_datetime(
            self.onset_df[self.onset_col],
            errors="coerce",
        )

    def trim_onset_columns(self):
        """
        Keep only columns needed for plotting before merging into the long table.
        """
        self.onset_df = self.onset_df[[self.mrn_col, self.onset_col]].copy()

    def merge_data(self):
        """
        Merge extracted review data with onset data by MRN.
        """
        self.merged_df = pd.merge(
            self.review_df,
            self.onset_df,
            on=self.mrn_col,
            how="inner",
        )

    def drop_missing_dates(self):
        """
        Remove rows missing either review date or onset date.
        """
        self.merged_df = self.merged_df.dropna(
            subset=[self.date_col, self.onset_col]
        )

    def filter_invalid_dates(self):
        """
        Remove dates outside valid clinical/documentation range.
        """
        date_year = self.merged_df[self.date_col].dt.year
        onset_year = self.merged_df[self.onset_col].dt.year

        self.merged_df = self.merged_df[
            (date_year > self.min_valid_year)
            & (date_year < self.max_valid_year)
            & (onset_year > self.min_valid_year)
            & (onset_year < self.max_valid_year)
        ]

    def compute_months_since_onset(self):
        """
        Compute months between note/document date and onset date.
        """
        self.merged_df["months_since_onset"] = (
            self.merged_df[self.date_col] - self.merged_df[self.onset_col]
        ).dt.days / 30.44

    def build_symptom_series(self):
        """
        Pre-aggregate plotted series once. This avoids repeatedly slicing and
        grouping the same long dataframe for every MRN/symptom pair.
        """
        if {"Question", "Status"}.issubset(self.merged_df.columns):
            self.series_df = self.build_long_format_series()
        else:
            self.series_df = self.build_wide_format_series()

    def build_long_format_series(self):
        """
        Build exact-key long-format symptom trajectories.
        """
        df = self.merged_df[self.merged_df["Question"].isin(self.symptom_keys)].copy()
        if df.empty:
            return pd.DataFrame(columns=[self.mrn_col, "Question", "months_since_onset", "Status"])

        df["Status"] = pd.to_numeric(df["Status"], errors="coerce")
        df = df.dropna(subset=["Status"])

        return (
            df.groupby([self.mrn_col, "Question", "months_since_onset"], as_index=False)["Status"]
            .max()
            .sort_values([self.mrn_col, "Question", "months_since_onset"])
        )

    def build_wide_format_series(self):
        """
        Build trajectories from wide-format symptom columns.
        """
        records = []
        available_keys = [key for key in self.symptom_keys if key in self.merged_df.columns]

        for key in available_keys:
            symptom_df = self.merged_df[[self.mrn_col, "months_since_onset", key]].copy()
            symptom_df[key] = pd.to_numeric(symptom_df[key], errors="coerce")
            symptom_df = symptom_df.dropna(subset=[key])
            if symptom_df.empty:
                continue

            symptom_df = (
                symptom_df
                .groupby([self.mrn_col, "months_since_onset"], as_index=False)[key]
                .max()
                .rename(columns={key: "Status"})
            )
            symptom_df["Question"] = key
            records.append(symptom_df)

        if not records:
            return pd.DataFrame(columns=[self.mrn_col, "Question", "months_since_onset", "Status"])

        return pd.concat(records, ignore_index=True).sort_values(
            [self.mrn_col, "Question", "months_since_onset"]
        )

    def plot_all(self):
        """
        Generate one combined Plotly figure. Every plotted patient/symptom
        trajectory is a trace on that single figure.
        """
        self.validate_symptom_keys()
        return self.plot_combined()

    def plot_combined(self):
        """
        Plot all requested symptom keys into one HTML file.
        """
        fig = go.Figure()
        plotted_any = False
        shown_trace_classes = set()
        
        # Pre-calculate incidence for legend annotations
        incidence_map = self._calculate_incidence_map()

        for symptom_key in self.symptom_keys:
            symptom_df = self.series_df[self.series_df["Question"] == symptom_key]
            if symptom_df.empty:
                print(f"Warning: symptom key '{symptom_key}' has no plotted data.")
                continue

            for mrn, patient_df in symptom_df.groupby(self.mrn_col):
                patient_df = self.prepare_trace_dataframe(patient_df)
                if self.drop_unknown:
                    patient_df = patient_df[patient_df["State"] != -1]
                    if patient_df.empty:
                        continue

                trace_class = self.classify_trace(patient_df, self.transition_threshold_months)
                if not self.enabled_trace_conditions[trace_class]:
                    continue

                # Only show legend entry for first trace of each class
                show_legend = trace_class not in shown_trace_classes
                if show_legend:
                    shown_trace_classes.add(trace_class)

                y_display = patient_df["State"] + self.TRACE_Y_OFFSETS[trace_class]
                color = self.TRACE_COLORS[trace_class]
                
                # Build legend name with incidence if available
                legend_name = trace_class
                if trace_class in incidence_map:
                    incidence_pct = incidence_map[trace_class] * 100
                    legend_name = f"{trace_class} ({self.transition_threshold_months} Month Incidence: {incidence_pct:.1f}%)"
                else:
                    incidence_pct = 0
                    legend_name = f"{trace_class} ({self.transition_threshold_months} Month Incidence: {incidence_pct:.1f}%)"
                    
                
                fig.add_trace(
                    go.Scatter(
                        x=patient_df["months_since_onset"],
                        y=y_display,
                        mode="lines+markers",
                        line={"shape": "hv", "color": color},
                        marker={"color": color, "size": 5},
                        name=legend_name,
                        legendgroup=trace_class,
                        showlegend=show_legend,
                        hovertemplate=(
                            "MRN=%{customdata[0]}<br>"
                            "Symptom=%{customdata[1]}<br>"
                            "Months from onset=%{x:.2f}<br>"
                            "State=%{customdata[2]}<br>"
                            f"Class={trace_class}<extra></extra>"
                        ),
                        customdata=[
                            [str(mrn), symptom_key, state]
                            for state in patient_df["StateLabel"]
                        ],
                    )
                )
                plotted_any = True

        if not plotted_any:
            raise ValueError("No plotted symptom trajectories were available.")

        self.format_figure(fig, self.series_df)
        output_path = self.get_output_filepath()
        fig.write_html(output_path, include_plotlyjs="cdn")
        return fig, output_path
    
    def _calculate_incidence_map(self):
        """
        Helper method to calculate incidence rates for each transition type.
        Observes occurrences across ±12-month window (default x_limit) and converts
        to transition_threshold_months incidence rate.
        Returns a dict mapping trace_class to normalized incidence ratio.
        """
        incidence_map = {}
        transition_types = ["NO -> YES", "YES -> NO", "NO -> NO", "YES -> YES"]
        x_limit = 12  # Default observation window
        
        for symptom_key in self.symptom_keys:
            symptom_df = self.series_df[self.series_df["Question"] == symptom_key]
            if symptom_df.empty:
                continue
            
            # Initialize counters
            counts = {t: 0 for t in transition_types}
            total_patients = 0
            
            for mrn, patient_df in symptom_df.groupby(self.mrn_col):
                patient_df = self.prepare_trace_dataframe(patient_df)
                if self.drop_unknown:
                    patient_df = patient_df[patient_df["State"] != -1]
                
                if patient_df.empty:
                    continue
                
                total_patients += 1
                
                # Filter to ±12 month observation window
                obs_window = patient_df[
                    (patient_df["months_since_onset"] >= -x_limit) &
                    (patient_df["months_since_onset"] <= x_limit)
                ]
                
                if obs_window.empty:
                    continue
                
                # Determine state before and after onset within window
                before = obs_window[obs_window["months_since_onset"] < 0]
                after = obs_window[obs_window["months_since_onset"] >= 0]
                
                before_state = before["State"].max() if not before.empty else -1
                after_state = after["State"].max() if not after.empty else -1
                
                # Map states to YES/NO
                before_yes = (before_state == 1)
                after_yes = (after_state == 1)
                
                # Classify transition
                if before_yes and after_yes:
                    transition = "YES -> YES"
                elif before_yes and not after_yes:
                    transition = "YES -> NO"
                elif not before_yes and after_yes:
                    transition = "NO -> YES"
                else:
                    transition = "NO -> NO"
                
                counts[transition] += 1
            
            # Convert ±12-month incidence to transition_threshold_months incidence
            if total_patients > 0:
                incidence_window_ratio = self.transition_threshold_months / x_limit
                for transition_type, transition_name in [
                    ("NO -> YES", "NO BEFORE ONSET | YES AFTER ONSET"),
                    ("YES -> NO", "YES BEFORE ONSET | NO AFTER ONSET"),
                    ("NO -> NO", "NO BEFORE ONSET | NO AFTER ONSET"),
                    ("YES -> YES", "YES BEFORE ONSET | YES AFTER ONSET"),
                ]:
                    incidence_x_window = counts[transition_type] / total_patients
                    incidence_threshold_window = incidence_x_window * incidence_window_ratio
                    incidence_map[transition_name] = incidence_threshold_window
        
        return incidence_map

    @staticmethod
    def display_figure(fig):
        """
        Display the Plotly figure inline when running in IPython/Jupyter.
        """
        try:
            from IPython import get_ipython
            from IPython.display import display
        except ImportError:
            return

        shell = get_ipython()
        if shell is None:
            return

        display(fig)

    @classmethod
    def prepare_trace_dataframe(cls, patient_df):
        """
        Sort one trajectory and apply symptom timing carry-forward rules.

        Raw ReviewPyper status is mapped as:
          - 2 -> 1  (Yes)
          - 1 -> 0  (No)
          - 0 -> -1 (No data)

        Then:
          - once Yes occurs, all later points remain Yes
          - before Yes, No carries forward over later No Data
        """
        patient_df = patient_df.sort_values("months_since_onset").copy()
        patient_df["State"] = cls.apply_state_rules(patient_df["Status"])
        patient_df["StateLabel"] = patient_df["State"].map({
            -1: "No data",
            0: "No",
            1: "Yes",
        })
        return patient_df

    @staticmethod
    def apply_state_rules(status_series):
        """
        Apply the state machine for symptom timing.
        """
        state = -1
        states = []

        for raw_status in status_series:
            if raw_status == 2:
                raw_state = 1
            elif raw_status == 1:
                raw_state = 0
            else:
                raw_state = -1

            if state == 1:
                pass
            elif raw_state == 1:
                state = 1
            elif raw_state == 0:
                state = 0

            states.append(state)

        return states

    @staticmethod
    def classify_trace(patient_df, transition_threshold_months=3):
        """
        Classify a trajectory by whether Yes appears before and after onset.
        
        If a transition occurs (no->yes or yes->no), only counts if the transition
        occurs within transition_threshold_months of onset.
        
        Args:
            patient_df: DataFrame with State and months_since_onset columns
            transition_threshold_months: Maximum months from onset for a transition to count
        """
        before_df = patient_df[patient_df["months_since_onset"] < 0]
        after_df = patient_df[patient_df["months_since_onset"] >= 0]

        before_yes = bool((before_df["State"] == 1).any())
        after_yes = bool((after_df["State"] == 1).any())

        # If there's a transition after onset, check if it occurs within threshold
        if before_yes and not after_yes:
            # Transition from YES to NO - check if NO occurs within threshold
            no_after = after_df[after_df["State"] == 0]
            if not no_after.empty:
                first_no_time = no_after["months_since_onset"].min()
                if first_no_time > transition_threshold_months:
                    # Transition occurs too late, treat as still YES after onset
                    after_yes = True
        
        elif not before_yes and after_yes:
            # Transition from NO to YES - check if YES occurs within threshold
            yes_after = after_df[after_df["State"] == 1]
            if not yes_after.empty:
                first_yes_time = yes_after["months_since_onset"].min()
                if first_yes_time > transition_threshold_months:
                    # Transition occurs too late, treat as still NO after onset
                    after_yes = False

        before_label = "YES BEFORE ONSET" if before_yes else "NO BEFORE ONSET"
        after_label = "YES AFTER ONSET" if after_yes else "NO AFTER ONSET"
        return f"{before_label} | {after_label}"

    def validate_symptom_keys(self):
        """
        Ensure requested symptom keys exactly match keys present in the JSON-derived
        longitudinal data.
        """
        source_df = self.series_df if self.series_df is not None else self.merged_df
        if source_df is None or "Question" not in source_df.columns:
            return

        available_keys = sorted(source_df["Question"].dropna().astype(str).unique())
        missing = [key for key in self.symptom_keys if key not in available_keys]

        if not missing:
            return

        available = "\n".join(f"  - {key}" for key in available_keys)
        raise ValueError(
            "Requested symptom keys were not found in the ReviewPyper JSON-derived "
            f"data: {missing}\nAvailable symptom keys:\n{available}"
        )

    def format_figure(self, fig, symptom_df):
        """
        Apply Plotly layout, labels, and onset reference line.
        """
        # Create dynamic labels from onset column name
        onset_label = self.onset_col.replace("_", " ").title()
        months_label = f"Months from {onset_label}"
        
        fig.add_vline(
            x=0,
            line_width=2,
            line_color="black",
            annotation_text=onset_label,
            annotation_position="top",
        )

        fig.update_layout(
            title="Symptom Progression",
            xaxis_title=months_label,
            yaxis_title="Symptom Status",
            xaxis={
                "range": [-12, 12],
                "zeroline": False,
                "showgrid": False,
            },
            yaxis={
                **self.get_yaxis_config(),
                "showgrid": False,
            },
            hovermode="closest",
            showlegend=True,
            legend={
                "x": 1.02,
                "y": 1.0,
                "xanchor": "left",
                "yanchor": "top",
                "bgcolor": "rgba(255, 255, 255, 0.9)",
                "xref": "paper",
                "yref": "paper",
            },
            template="plotly_white",
        )

    def get_yaxis_config(self):
        """
        Build y-axis ticks based on whether unknown states are hidden.
        """
        if self.drop_unknown:
            return {
                "range": [-0.4, 1.4],
                "tickmode": "array",
                "tickvals": [0, 1],
                "ticktext": ["0: No", "1: Yes"],
            }

        return {
            "range": [-1.4, 1.4],
            "tickmode": "array",
            "tickvals": [-1, 0, 1],
            "ticktext": ["-1: No Data", "0: No", "1: Yes"],
        }

    def get_output_filepath(self):
        """
        Build safe output filepath.
        """
        filename = "symptom_progression.html"

        return self.output_dir / filename

    @staticmethod
    def make_safe_filename_component(value):
        """
        Convert arbitrary MRN/symptom text into a safe filename component.
        """
        value = str(value).strip()
        value = re.sub(r"[^\w.-]+", "_", value)
        value = re.sub(r"_+", "_", value)
        value = value[:30]
        return value.strip("_")

    def calculate_transition_incidence(self, x_limit_months=12):
        """
        Calculate incidence rates by observing occurrences across x_limit_months window
        and converting to transition_threshold_months incidence rate.
        
        For each symptom, counts transitions in the ±x_limit window and scales to the
        transition_threshold_months timeframe (e.g., 3-month incidence).
        
        For each transition type:
        - NO -> YES (incidence of new symptom onset)
        - YES -> NO (resolution/remission of symptom)
        - NO -> NO (symptom never develops)
        - YES -> YES (symptom persists)
        
        Args:
            x_limit_months (float): Observation window in months (default: 12, means ±12 months)
            
        Returns:
            pd.DataFrame: Summary with columns [Symptom, Transition_Type, Count, Total_Patients, Incidence_in_X_months]
        """
        if self.series_df is None or self.series_df.empty:
            raise ValueError("No processed symptom data available. Run prepare_data() first.")
        
        records = []
        transition_types = ["NO -> YES", "YES -> NO", "NO -> NO", "YES -> YES"]
        
        for symptom_key in self.symptom_keys:
            symptom_df = self.series_df[self.series_df["Question"] == symptom_key]
            if symptom_df.empty:
                continue
            
            # Initialize counters
            counts = {t: 0 for t in transition_types}
            total_patients = 0
            
            for mrn, patient_df in symptom_df.groupby(self.mrn_col):
                patient_df = self.prepare_trace_dataframe(patient_df)
                if self.drop_unknown:
                    patient_df = patient_df[patient_df["State"] != -1]
                
                if patient_df.empty:
                    continue
                
                total_patients += 1
                
                # Filter to observation window
                obs_window = patient_df[
                    (patient_df["months_since_onset"] >= -x_limit_months) &
                    (patient_df["months_since_onset"] <= x_limit_months)
                ]
                
                if obs_window.empty:
                    continue
                
                # Determine state before and after onset within window
                before = obs_window[obs_window["months_since_onset"] < 0]
                after = obs_window[obs_window["months_since_onset"] >= 0]
                
                before_state = before["State"].max() if not before.empty else -1
                after_state = after["State"].max() if not after.empty else -1
                
                # Map states to YES/NO
                before_yes = (before_state == 1)
                after_yes = (after_state == 1)
                
                # Classify transition
                if before_yes and after_yes:
                    transition = "YES -> YES"
                elif before_yes and not after_yes:
                    transition = "YES -> NO"
                elif not before_yes and after_yes:
                    transition = "NO -> YES"
                else:
                    transition = "NO -> NO"
                
                counts[transition] += 1
            
            # Record incidence for each transition type (even if 0)
            if total_patients > 0:
                for transition in transition_types:
                    incidence_x_window = counts[transition] / total_patients
                    # Convert to transition_threshold_months incidence
                    incidence_threshold_window = incidence_x_window * (self.transition_threshold_months / x_limit_months)
                    
                    records.append({
                        "Symptom": symptom_key,
                        "Transition_Type": transition,
                        "Count": counts[transition],
                        "Total_Patients": total_patients,
                        f"Incidence_in_{int(self.transition_threshold_months)}_months": round(incidence_threshold_window, 3),
                    })
        
        incidence_df = pd.DataFrame(records)
        
        # Export to CSV
        output_path = self.output_dir / "transition_incidence.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        incidence_df.to_csv(output_path, index=False)
        print(f"Transition incidence summary exported to {output_path}")
        
        return incidence_df

    def export_csv(self, output_path=None):
        """
        Export patient identifiers and their symptom condition classifications to CSV.

        For each patient-symptom pair, determines the trace classification
        (e.g., "YES BEFORE ONSET | YES AFTER ONSET") and writes to CSV.

        Args:
            output_path (str or Path, optional): Path to output CSV file.
                If not provided, uses output_dir/patient_conditions.csv

        Returns:
            pd.DataFrame: The exported dataframe
        """
        if self.series_df is None or self.series_df.empty:
            raise ValueError("No processed symptom data available. Run prepare_data() first.")

        if output_path is None:
            output_path = self.output_dir / "patient_conditions.csv"
        else:
            output_path = Path(output_path)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build export dataframe: patient ID and condition for each symptom
        records = []
        for symptom_key in self.symptom_keys:
            symptom_df = self.series_df[self.series_df["Question"] == symptom_key]
            if symptom_df.empty:
                continue

            for mrn, patient_df in symptom_df.groupby(self.mrn_col):
                patient_df = self.prepare_trace_dataframe(patient_df)
                if self.drop_unknown:
                    patient_df = patient_df[patient_df["State"] != -1]
                    if patient_df.empty:
                        continue

                condition = self.classify_trace(patient_df, self.transition_threshold_months)
                records.append({
                    self.mrn_col: mrn,
                    "Symptom": symptom_key,
                    "Condition": condition,
                })

        export_df = pd.DataFrame(records)

        if export_df.empty:
            raise ValueError("No patient-symptom records to export.")

        export_df.to_csv(output_path, index=False)
        print(f"Patient conditions exported to {output_path}")
        return export_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate patient symptom progression step plots from ReviewPyper JSON."
    )

    parser.add_argument(
        "--json",
        required=True,
        help="Path to ReviewPyper output JSON.",
    )

    parser.add_argument(
        "--onset",
        required=True,
        help="Path to onset CSV.",
    )

    parser.add_argument(
        "--onset_col",
        default="stroke_date",
        help="Column name in onset CSV containing stroke/onset date.",
    )

    parser.add_argument(
        "--symptoms",
        required=True,
        nargs="+",
        help="One or more exact ReviewPyper JSON symptom/question keys to plot.",
    )

    parser.add_argument(
        "--outdir",
        default="dateTimePlots",
        help="Output directory for plots.",
    )

    parser.add_argument(
        "--drop_unknown",
        action="store_true",
        help="Hide -1/no-data portions of traces after applying carry-forward rules.",
    )

    parser.add_argument(
        "--hide_yes_before_yes_after",
        action="store_true",
        help="Hide traces classified as YES BEFORE ONSET | YES AFTER ONSET.",
    )

    parser.add_argument(
        "--hide_yes_before_no_after",
        action="store_true",
        help="Hide traces classified as YES BEFORE ONSET | NO AFTER ONSET.",
    )

    parser.add_argument(
        "--hide_no_before_no_after",
        action="store_true",
        help="Hide traces classified as NO BEFORE ONSET | NO AFTER ONSET.",
    )

    parser.add_argument(
        "--hide_no_before_yes_after",
        action="store_true",
        help="Hide traces classified as NO BEFORE ONSET | YES AFTER ONSET.",
    )

    parser.add_argument(
        "--no_display",
        action="store_true",
        help="Do not display the Plotly figure inline when running in a notebook.",
    )

    return parser.parse_args()

if __name__ == "__main__":
    json_path="/Users/cu135/hires_backdrops/t/emr_strict_extraction_evaluations.json"  # must have symptoms for a patients response from review pyper
    onset_csv_path="/Users/cu135/Downloads/hbs_depression_evaluated.csv"               # must have an onset date to register to
    symptoms=["Does this patient have difficulty with memory, such as being unable to remember a list of words read to them?"]        # This is a list of the JSON keys to extract symptoms from
    output_dir="/Users/cu135/hires_backdrops/t"    
    onset_col="stroke_date"        # From the CSV, determines 'onset' to lock each patient's trace to
    drop_unknown=True              # This removes 'unknown' from showing up on the plot
    transition_threshold_months=3  # This deteremines within how many months from onset we consider the onset causing the symptom change

    plotter = SymptomProgressionPlotter(
        json_path=json_path,
        onset_csv_path=onset_csv_path,
        symptoms=symptoms,
        output_dir=output_dir,
        onset_col=onset_col,
        drop_unknown=drop_unknown,
        transition_threshold_months=transition_threshold_months
    )
    plotter.run()
    plotter.export_csv()
    
    # Calculate and export transition incidence
    incidence_summary = plotter.calculate_transition_incidence(x_limit_months=12)
    print(f"\n=== TRANSITION INCIDENCE SUMMARY ===")
    print(f"Observation window: ±12 months")
    print(f"Reporting incidence in: ±{transition_threshold_months}-month window")
    print(incidence_summary.to_string(index=False))
    print(f"\nIncidence = (Count observed in ±12 months / Total Patients) × ({transition_threshold_months} / 12)")