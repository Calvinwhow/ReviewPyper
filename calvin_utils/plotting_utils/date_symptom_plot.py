import argparse
from numbers import Real
import re
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from calvin_utils.gpt_sys_review.gpt_utils.temporal_analysis import TemporalPlotter


def normalize_transition_window(transition_threshold_months):
    """
    Normalize scalar or (lower, upper) transition-window configuration.
    """
    if isinstance(transition_threshold_months, (tuple, list)):
        if len(transition_threshold_months) != 2:
            raise ValueError("transition_threshold_months must be a scalar or (lower, upper).")
        lower, upper = transition_threshold_months
    else:
        lower, upper = 0, transition_threshold_months

    if not isinstance(lower, Real) or not isinstance(upper, Real):
        raise TypeError("transition window bounds must be numeric.")
    if lower < 0:
        raise ValueError("transition lower limit must be >= 0.")
    if upper < lower:
        raise ValueError("transition upper limit must be >= lower limit.")

    return float(lower), float(upper)


def format_transition_window_label(transition_window_months):
    """
    Format transition window bounds for legends and output column names.
    """
    lower, upper = transition_window_months
    lower_label = f"{lower:g}"
    upper_label = f"{upper:g}"
    if lower == 0:
        return upper_label
    return f"{lower_label}-{upper_label}"


class SymptomPlottingStatusCalculator:
    """
    Convert ReviewPyper statuses into plotting state values.
    """
    VALID_RULES = {"raw", "hierarchical", "coerce", "average"}

    STATE_LABELS = {
        -1: "No data",
        0: "No",
        1: "Yes",
    }

    def __init__(
        self,
        state_rule="hierarchical",
        floor_threshold=None,
        coerce_observation_threshold=3,
        transition_threshold_months=3,
    ):
        if state_rule not in self.VALID_RULES:
            valid = ", ".join(sorted(self.VALID_RULES))
            raise ValueError(f"state_rule must be one of: {valid}")

        self.state_rule = state_rule
        self.floor_threshold = floor_threshold
        self.coerce_observation_threshold = coerce_observation_threshold
        self.set_transition_threshold_months(transition_threshold_months)

    def set_transition_threshold_months(self, transition_threshold_months):
        """
        Store a backward-compatible transition threshold plus explicit bounds.
        """
        lower, upper = normalize_transition_window(transition_threshold_months)
        self.transition_lower_limit_months = lower
        self.transition_upper_limit_months = upper
        self.transition_threshold_months = upper
        self.transition_window_months = (lower, upper)

    def is_in_transition_window(self, month):
        """
        Return whether month is inside the configured post-onset window.
        """
        return (
            month is not None
            and self.transition_lower_limit_months <= month <= self.transition_upper_limit_months
        )

    @staticmethod
    def raw_status_to_state(raw_status):
        """
        Map raw ReviewPyper status into plot state.

        ReviewPyper status:
          - 2 -> 1  (Yes)
          - 1 -> 0  (No)
          - any other value -> -1 (No data)
        """
        if raw_status == 2:
            return 1
        if raw_status == 1:
            return 0
        return -1

    def raw_states(self, status_series):
        """
        Convert raw statuses directly to plotted state values.
        """
        return [self.raw_status_to_state(raw_status) for raw_status in status_series]

    def coerce_states(self, status_series):
        """
        Change the plotted state only after more than the configured number of
        consecutive observations support the new Yes/No state.
        """
        state = -1
        candidate_state = None
        candidate_count = 0
        states = []

        for raw_status in status_series:
            raw_state = self.raw_status_to_state(raw_status)

            if raw_state == -1:
                states.append(state)
                continue

            if state == -1:
                state = raw_state
                candidate_state = None
                candidate_count = 0
            elif raw_state == state:
                candidate_state = None
                candidate_count = 0
            else:
                if raw_state == candidate_state:
                    candidate_count += 1
                else:
                    candidate_state = raw_state
                    candidate_count = 1

                if candidate_count > self.coerce_observation_threshold:
                    state = candidate_state
                    candidate_state = None
                    candidate_count = 0

            states.append(state)

        return states

    def hierarchical_states(self, status_series, months_since_onset=None):
        """
        Plot symptom state over time. For baseline-Yes patients, any No inside
        the onset threshold window locks the trajectory to No permanently.
        For baseline-No patients, any Yes inside the onset threshold window
        locks the trajectory to Yes permanently.
        """
        raw_states = self.raw_states(status_series)
        states = []

        if months_since_onset is None:
            months_since_onset = [None] * len(status_series)

        baseline_state = self.baseline_state(raw_states, months_since_onset)
        conversion_index = self.threshold_conversion_index(raw_states, months_since_onset, baseline_state)
        state = -1
        locked_state = None

        for index, (raw_state, month) in enumerate(zip(raw_states, months_since_onset)):
            if locked_state is not None:
                states.append(locked_state)
                continue

            if raw_state == -1:
                states.append(state)
                continue

            if state == -1:
                state = raw_state
            elif conversion_index is not None and index == conversion_index:
                state = raw_state
                if baseline_state == 1 and raw_state == 0:
                    locked_state = 0
                elif baseline_state == 0 and raw_state == 1:
                    locked_state = 1
            elif raw_state == 1:
                state = 1
            elif month is not None and month <= self.transition_upper_limit_months and state != 1:
                state = raw_state

            states.append(state)

        return states

    @staticmethod
    def baseline_state(raw_states, months_since_onset):
        """
        Before onset, any Yes means baseline Yes. Otherwise baseline is No.
        """
        for raw_state, month in zip(raw_states, months_since_onset):
            if month is not None and month < 0 and raw_state == 1:
                return 1
        return 0

    def threshold_conversion_index(self, raw_states, months_since_onset, baseline_state):
        """
        Return the first index where a patient converts away from baseline
        during the post-onset threshold window.
        """
        target_state = 0 if baseline_state == 1 else 1
        for index, (raw_state, month) in enumerate(zip(raw_states, months_since_onset)):
            if (
                self.is_in_transition_window(month)
                and raw_state == target_state
            ):
                return index
        return None

    def is_threshold_transition(self, state, raw_state, month):
        """
        Return whether raw_state is a real Yes/No transition within the
        configured post-onset threshold window.
        """
        return (
            self.is_in_transition_window(month)
            and state in {0, 1}
            and raw_state in {0, 1}
            and raw_state != state
        )

    def average_states(self, status_series):
        """
        Convert raw statuses to states, then average each point with its
        immediate previous and next point when available.
        """
        states = self.raw_states(status_series)
        averaged_states = []

        for index, state in enumerate(states):
            window = states[max(0, index - 1):index + 2]
            averaged_states.append(sum(window) / len(window))

        return averaged_states

    def calculate_states(self, status_series, months_since_onset=None):
        """
        Return plotted state values using the configured state rule.
        """
        if self.state_rule == "raw":
            states = self.raw_states(status_series)
        elif self.state_rule == "hierarchical":
            states = self.hierarchical_states(status_series, months_since_onset)
        elif self.state_rule == "average":
            states = self.average_states(status_series)
        else:
            states = self.coerce_states(status_series)

        return self.apply_floor_threshold(states)

    def apply_floor_threshold(self, states):
        """
        Set values below floor_threshold to 0.
        """
        if self.floor_threshold is None:
            return states

        return [
            0 if state < self.floor_threshold else state
            for state in states
        ]

    def prepare_trace_dataframe(self, patient_df):
        """
        Sort one trajectory and add plotting State and StateLabel columns.
        """
        patient_df = patient_df.sort_values("months_since_onset").copy()
        patient_df["RawState"] = self.raw_states(patient_df["Status"])
        patient_df["State"] = self.calculate_states(
            patient_df["Status"],
            patient_df["months_since_onset"],
        )
        patient_df["StateLabel"] = patient_df["State"].map(self.STATE_LABELS)
        patient_df["StateLabel"] = patient_df["StateLabel"].fillna(
            patient_df["State"].map(lambda state: f"{state:.2f}")
        )
        return patient_df


class SymptomPlottingConditions:
    """
    Classify plotted symptom trajectories from already-calculated state values.
    """
    TRACE_TRANSITIONS = {
        "NO -> YES": "NO BEFORE ONSET | YES AFTER ONSET",
        "YES -> NO": "YES BEFORE ONSET | NO AFTER ONSET",
        "NO -> NO": "NO BEFORE ONSET | NO AFTER ONSET",
        "YES -> YES": "YES BEFORE ONSET | YES AFTER ONSET",
    }
    TRANSITION_TYPES = list(TRACE_TRANSITIONS.keys())

    @staticmethod
    def final_state(df):
        """
        Return the last plotted state in a trajectory segment.
        """
        if df.empty:
            return -1
        return df.sort_values("months_since_onset")["State"].iloc[-1]

    @classmethod
    def classify_trace(cls, patient_df, transition_threshold_months=3):
        """
        Classify a trajectory from baseline-before-onset and the post-onset
        threshold window only. Later symptom changes do not alter class/color.
        """
        state_col = "RawState" if "RawState" in patient_df.columns else "State"
        lower_limit, upper_limit = normalize_transition_window(transition_threshold_months)
        before_df = patient_df[patient_df["months_since_onset"] < 0]
        window_df = patient_df[
            (patient_df["months_since_onset"] >= lower_limit)
            & (patient_df["months_since_onset"] <= upper_limit)
        ]

        before_yes = bool((before_df[state_col] == 1).any())
        if before_yes:
            after_yes = not bool((window_df[state_col] == 0).any())
        else:
            after_yes = bool((window_df[state_col] == 1).any())

        before_label = "YES BEFORE ONSET" if before_yes else "NO BEFORE ONSET"
        after_label = "YES AFTER ONSET" if after_yes else "NO AFTER ONSET"
        return f"{before_label} | {after_label}"

    @staticmethod
    def states_to_transition(before_state, after_state):
        """
        Return the transition label for before/after onset states.
        """
        before_yes = before_state == 1
        after_yes = after_state == 1

        if before_yes and after_yes:
            return "YES -> YES"
        if before_yes and not after_yes:
            return "YES -> NO"
        if not before_yes and after_yes:
            return "NO -> YES"
        return "NO -> NO"

    @classmethod
    def classify_window_transition(cls, patient_df, x_limit_months):
        """
        Return transition type within a symmetric onset observation window.
        """
        obs_window = patient_df[
            (patient_df["months_since_onset"] >= -x_limit_months)
            & (patient_df["months_since_onset"] <= x_limit_months)
        ]

        if obs_window.empty:
            return None

        before = obs_window[obs_window["months_since_onset"] < 0]
        after = obs_window[obs_window["months_since_onset"] >= 0]

        state_col = "RawState" if "RawState" in patient_df.columns else "State"
        before_state = 1 if (before[state_col] == 1).any() else 0
        if before_state == 1:
            after_state = 0 if (after[state_col] == 0).any() else 1
        else:
            after_state = 1 if (after[state_col] == 1).any() else 0
        return cls.states_to_transition(before_state, after_state)

    @classmethod
    def transition_to_trace_class(cls, transition):
        """
        Return the plotted trace class for a transition label.
        """
        return cls.TRACE_TRANSITIONS[transition]


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
        "NO BEFORE ONSET | YES AFTER ONSET (Non-causal)": "#d62728",
    }
    TRACE_Y_OFFSETS = {
        "YES BEFORE ONSET | YES AFTER ONSET": 0.075,
        "YES BEFORE ONSET | NO AFTER ONSET": 0.025,
        "NO BEFORE ONSET | NO AFTER ONSET": -0.075,
        "NO BEFORE ONSET | YES AFTER ONSET": 0.025,
        "NO BEFORE ONSET | YES AFTER ONSET (Non-causal)": -0.025,
    }

    def __init__(
        self,
        json_path,
        onset_csv_path,
        symptoms,
        question_key_file=None,
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
        show_no_before_non_causal_after=True,
        display_plot=True,
        transition_threshold_months=3,
        state_rule="hierarchical",
        state_floor_threshold=None,
        coerce_observation_threshold=3,
    ):
        self.json_path = Path(json_path)
        self.onset_csv_path = Path(onset_csv_path)
        self.requested_symptom_keys = list(symptoms)
        self.symptom_keys = list(symptoms)
        self.symptom_key_map = {}
        self.question_key_file = Path(question_key_file) if question_key_file is not None else None
        self.question_names_dict = None
        self.output_dir = Path(output_dir)
        self.requested_onset_col = onset_col
        self.onset_col = onset_col
        self.date_col = date_col
        self.mrn_col = mrn_col
        self.temp_longitudinal_path = self.output_dir / temp_longitudinal_filename
        self.min_valid_year = min_valid_year
        self.max_valid_year = max_valid_year
        self.drop_unknown = drop_unknown
        self.set_transition_threshold_months(transition_threshold_months)
        self.state_rule = state_rule
        self.state_floor_threshold = state_floor_threshold
        self.coerce_observation_threshold = coerce_observation_threshold
        self.enabled_trace_conditions = {
            "YES BEFORE ONSET | YES AFTER ONSET": show_yes_before_yes_after,
            "YES BEFORE ONSET | NO AFTER ONSET": show_yes_before_no_after,
            "NO BEFORE ONSET | NO AFTER ONSET": show_no_before_no_after,
            "NO BEFORE ONSET | YES AFTER ONSET": show_no_before_yes_after,
            "NO BEFORE ONSET | YES AFTER ONSET (Non-causal)": show_no_before_non_causal_after,
        }
        self.display_plot = display_plot
        self.status_calculator = SymptomPlottingStatusCalculator(
            state_rule,
            floor_threshold=state_floor_threshold,
            coerce_observation_threshold=coerce_observation_threshold,
            transition_threshold_months=self.transition_window_months,
        )
        self.plotting_conditions = SymptomPlottingConditions()

        self.review_df = None
        self.onset_df = None
        self.merged_df = None
        self.series_df = None

    def set_transition_threshold_months(self, transition_threshold_months):
        """
        Store a backward-compatible transition threshold plus explicit bounds.
        """
        lower, upper = normalize_transition_window(transition_threshold_months)
        self.transition_lower_limit_months = lower
        self.transition_upper_limit_months = upper
        self.transition_threshold_months = upper
        self.transition_window_months = (lower, upper)

    def transition_window_label(self):
        """
        Return the transition window as a compact display label.
        """
        return format_transition_window_label(self.transition_window_months)

    def get_symptom_display_name(self, symptom_key):
        """
        Return a display/output label for a symptom key when a label map exists.
        """
        if self.question_names_dict:
            return self.question_names_dict.get(symptom_key, symptom_key)
        return symptom_key

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
        self.resolve_json_symptom_keys(plotter.data)

        self.review_df = plotter.export_per_date_longitudinal_data(
            str(self.temp_longitudinal_path),
            symptom_keys=self.symptom_keys,
        )
        if self.review_df is None or self.review_df.empty:
            raise ValueError("No valid longitudinal data could be extracted from JSON.")

    def resolve_json_symptom_keys(self, data):
        """
        Resolve requested symptom strings against evaluated JSON keys. Requests
        can be exact keys or unique substrings of a full key.
        """
        available_keys = self.get_json_symptom_keys(data)
        resolved_keys = []
        symptom_key_map = {}

        for requested_key in self.requested_symptom_keys:
            resolved_key = self.resolve_symptom_key(requested_key, available_keys)
            resolved_keys.append(resolved_key)
            symptom_key_map[requested_key] = resolved_key

            if resolved_key != requested_key:
                print(f"Resolved symptom substring to JSON key:\n  {requested_key}\n  -> {resolved_key}")

        self.symptom_keys = resolved_keys
        self.symptom_key_map = symptom_key_map

    @classmethod
    def get_json_symptom_keys(cls, data):
        """
        Return symptom/question keys available in ReviewPyper evaluated JSON.
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

        return available_keys

    @staticmethod
    def resolve_symptom_key(requested_key, available_keys):
        """
        Resolve one requested symptom key by exact match or unique substring.
        """
        if requested_key in available_keys:
            return requested_key

        matches = [
            available_key for available_key in available_keys
            if requested_key in available_key
        ]

        if len(matches) == 1:
            return matches[0]

        available = "\n".join(f"  - {key}" for key in sorted(available_keys))
        if not matches:
            raise ValueError(
                "Requested symptom key was not found as an exact key or substring "
                f"in the evaluated JSON: {requested_key}\nAvailable symptom keys:\n{available}"
            )

        matched = "\n".join(f"  - {key}" for key in sorted(matches))
        raise ValueError(
            "Requested symptom substring matched multiple JSON keys. Use a more "
            f"specific substring.\nRequested: {requested_key}\nMatches:\n{matched}"
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
        available_keys = []
        for key in self.symptom_keys:
            try:
                available_keys.append(self.resolve_symptom_key(key, set(self.merged_df.columns)))
            except ValueError:
                continue

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
                patient_df = self.status_calculator.prepare_trace_dataframe(patient_df)
                if self.drop_unknown:
                    patient_df = patient_df[patient_df["State"] != -1]
                    if patient_df.empty:
                        continue

                trace_class = self.plotting_conditions.classify_trace(
                    patient_df,
                    self.transition_window_months,
                )
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
                    legend_name = f"{trace_class}\n({self.transition_window_label()} Month Incidence: {incidence_pct:.1f}%)"
                else:
                    incidence_pct = 0
                    legend_name = f"{trace_class}\n({self.transition_window_label()} Month Incidence: {incidence_pct:.1f}%)"
                    
                
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
        output_symptoms = [
            self.get_symptom_display_name(key)
            for key in self.symptom_keys
        ]
        output_key = output_symptoms[0] if len(output_symptoms) == 1 else "multiple_symptoms"
        output_path = self.get_output_filepath("DateSymptomPlot", output_key, ".html")
        fig.write_html(output_path, include_plotlyjs="cdn")
        return fig, output_path
    
    def _calculate_incidence_map(self):
        """
        Calculate incidence rates using the onset-to-threshold classification
        window. No wider observation window or scaling is applied.
        Returns a dict mapping trace_class to normalized incidence ratio.
        """
        incidence_map = {}
        
        for symptom_key in self.symptom_keys:
            symptom_df = self.series_df[self.series_df["Question"] == symptom_key]
            if symptom_df.empty:
                continue
            
            counts = {
                trace_class: 0
                for trace_class in self.plotting_conditions.TRACE_TRANSITIONS.values()
            }
            total_patients = 0
            
            for mrn, patient_df in symptom_df.groupby(self.mrn_col):
                patient_df = self.status_calculator.prepare_trace_dataframe(patient_df)
                if self.drop_unknown:
                    patient_df = patient_df[patient_df["State"] != -1]
                
                if patient_df.empty:
                    continue
                
                total_patients += 1
                
                trace_class = self.plotting_conditions.classify_trace(
                    patient_df,
                    self.transition_window_months,
                )
                counts[trace_class] += 1
            
            if total_patients > 0:
                for trace_class, count in counts.items():
                    incidence_map[trace_class] = count / total_patients
        
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
        Sort one trajectory and map raw ReviewPyper status to plot state.

        Raw ReviewPyper status is mapped as:
          - 2 -> 1  (Yes)
          - 1 -> 0  (No)
          - 0 -> -1 (No data)
        """
        return SymptomPlottingStatusCalculator().prepare_trace_dataframe(patient_df)

    @staticmethod
    def apply_state_rules(status_series):
        """
        Legacy wrapper for the explicit coerce rule.
        """
        return SymptomPlottingStatusCalculator(state_rule="coerce").calculate_states(status_series)

    @staticmethod
    def classify_trace(patient_df, transition_threshold_months=3, remove_delayed_conversions=True):
        """
        Classify a trajectory by whether Yes appears before and after onset.
        
        If a transition occurs (no->yes or yes->no), only counts if the transition
        occurs within transition_threshold_months of onset.
        
        Args:
            patient_df: DataFrame with State and months_since_onset columns
            transition_threshold_months: Maximum months from onset for a transition to count
            remove_delayed_conversions: Whether to remove delayed conversions from the classification
        """
        return SymptomPlottingConditions.classify_trace(
            patient_df,
            transition_threshold_months,
        )

    def validate_symptom_keys(self):
        """
        Ensure requested symptom keys exactly match keys present in the JSON-derived
        longitudinal data.
        """
        source_df = self.series_df if self.series_df is not None else self.merged_df
        if source_df is None or "Question" not in source_df.columns:
            return

        available_keys = sorted(source_df["Question"].dropna().astype(str).unique())
        missing = []
        resolved_keys = []

        for key in self.symptom_keys:
            try:
                resolved_keys.append(self.resolve_symptom_key(key, set(available_keys)))
            except ValueError:
                missing.append(key)

        if not missing:
            self.symptom_keys = resolved_keys
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

    def get_output_filepath(self, prefix, symptom_key, ext):
        """
        Build safe output filepath.
        """
        filename = self.make_safe_filename_component(symptom_key)
        return self.output_dir / f"{prefix}_{filename}{ext}"

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

    def calculate_transition_incidence(self, x_limit_months=None):
        """
        Calculate transition incidence using the onset-to-threshold window.
        
        For each transition type:
        - NO -> YES (incidence of new symptom onset)
        - YES -> NO (resolution/remission of symptom)
        - NO -> NO (symptom never develops)
        - YES -> YES (symptom persists)
        
        Args:
            x_limit_months: Deprecated; retained for backward compatibility and ignored.
            
        Returns:
            pd.DataFrame: Summary with columns [Symptom, Transition_Type, Count, Total_Patients, Incidence_in_X_months]
        """
        if self.series_df is None or self.series_df.empty:
            raise ValueError("No processed symptom data available. Run prepare_data() first.")
        
        records = []
        transition_types = self.plotting_conditions.TRANSITION_TYPES
        trace_class_to_transition = {
            trace_class: transition
            for transition, trace_class in self.plotting_conditions.TRACE_TRANSITIONS.items()
        }
        
        for symptom_key in self.symptom_keys:
            symptom_df = self.series_df[self.series_df["Question"] == symptom_key]
            if symptom_df.empty:
                continue
            
            # Initialize counters
            counts = {t: 0 for t in transition_types}
            total_patients = 0
            
            for mrn, patient_df in symptom_df.groupby(self.mrn_col):
                patient_df = self.status_calculator.prepare_trace_dataframe(patient_df)
                if self.drop_unknown:
                    patient_df = patient_df[patient_df["State"] != -1]
                
                if patient_df.empty:
                    continue
                
                total_patients += 1
                
                trace_class = self.plotting_conditions.classify_trace(
                    patient_df,
                    self.transition_window_months,
                )
                transition = trace_class_to_transition[trace_class]

                counts[transition] += 1
            
            # Record incidence for each transition type (even if 0)
            if total_patients > 0:
                for transition in transition_types:
                    incidence_threshold_window = counts[transition] / total_patients
                    
                    records.append({
                        "Symptom": symptom_key,
                        "Transition_Type": transition,
                        "Count": counts[transition],
                        "Total_Patients": total_patients,
                        f"Incidence_in_{self.transition_window_label()}_months": round(incidence_threshold_window, 3),
                    })
        
        incidence_df = pd.DataFrame(records)
        
        # Export to CSV
        output_path = self.get_output_filepath("TransitionIncidence", symptom_key, ".csv")
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

        # Build export dataframe: patient ID and condition for each symptom
        records = []
        for symptom_key in self.symptom_keys:
            symptom_df = self.series_df[self.series_df["Question"] == symptom_key]
            if symptom_df.empty:
                continue

            for mrn, patient_df in symptom_df.groupby(self.mrn_col):
                patient_df = self.status_calculator.prepare_trace_dataframe(patient_df)
                if self.drop_unknown:
                    patient_df = patient_df[patient_df["State"] != -1]
                    if patient_df.empty:
                        continue

                condition = self.plotting_conditions.classify_trace(
                    patient_df,
                    self.transition_window_months,
                )
                records.append({
                    self.mrn_col: mrn,
                    "Symptom": symptom_key,
                    "Condition": condition,
                })

        export_df = pd.DataFrame(records)

        if export_df.empty:
            raise ValueError("No patient-symptom records to export.")
        
        export_df["Symptom"] = export_df["Symptom"].map(self.get_symptom_display_name)
        keys = [
            self.get_symptom_display_name(key)
            for key in self.symptom_keys
        ]
        
        if output_path is None:
            output_path = self.get_output_filepath("PatientConditions", keys, ".csv")
        
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
        help="Hide -1/no-data portions of traces after applying state rules.",
    )

    parser.add_argument(
        "--state_rule",
        default="hierarchical",
        choices=sorted(SymptomPlottingStatusCalculator.VALID_RULES),
        help=(
            "State calculation rule: raw leaves direct 1/0/-1 values; "
            "hierarchical carries forward the current state and accepts "
            "observed Yes->No or No->Yes transitions immediately; "
            "coerce changes state only after repeated observations support "
            "the new value; average averages each point with its immediate "
            "neighbors."
        ),
    )

    parser.add_argument(
        "--state_floor_threshold",
        type=float,
        default=None,
        help="Set calculated state values below this threshold to 0.",
    )

    parser.add_argument(
        "--coerce_observation_threshold",
        type=int,
        default=3,
        help=(
            "For state_rule=coerce, require more than this many consecutive "
            "observations before switching to a new Yes/No state."
        ),
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
        "--hide_no_before_non_causal_after",
        action="store_true",
        help="Hide traces classified as NO BEFORE ONSET | YES AFTER ONSET (Non-causal).",
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
    transition_threshold_months=1  # This deteremines within how many months from onset we consider the onset causing the symptom change
    state_rule="hierarchical"      # One of: raw, hierarchical, coerce, average
    state_floor_threshold=None     # Values below this threshold are set to 0
    coerce_observation_threshold=3 # Coerce changes state after >3 repeated observations

    plotter = SymptomProgressionPlotter(
        json_path=json_path,
        onset_csv_path=onset_csv_path,
        symptoms=symptoms,
        output_dir=output_dir,
        onset_col=onset_col,
        drop_unknown=drop_unknown,
        transition_threshold_months=transition_threshold_months,
        state_rule=state_rule,
        state_floor_threshold=state_floor_threshold,
        coerce_observation_threshold=coerce_observation_threshold,
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
