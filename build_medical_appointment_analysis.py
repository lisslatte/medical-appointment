from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from textwrap import dedent

import nbformat as nbf
import numpy as np
import pandas as pd
import plotly.express as px
from jinja2 import Template
from nbclient import NotebookClient
from plotly.offline import get_plotlyjs
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "noshowappointments-kagglev2-may-2016.csv"
OUTPUT_DIR = BASE_DIR / "analysis_outputs"
NOTEBOOK_FILE = BASE_DIR / "medical_no_show_analysis.ipynb"
DASHBOARD_FILE = BASE_DIR / "medical_no_show_story_dashboard.html"
INDEX_FILE = BASE_DIR / "index.html"

TARGET = "no_show"
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

FACTORS = OrderedDict(
    [
        ("wait_bucket", "Lead time"),
        ("age_group", "Age group"),
        ("appointment_weekday", "Appointment weekday"),
        ("scheduled_weekday", "Scheduled weekday"),
        ("gender", "Gender"),
        ("sms_received_label", "SMS reminder"),
        ("scholarship_label", "Scholarship / welfare"),
        ("condition_count_bucket", "Condition count"),
        ("hypertension_label", "Hypertension"),
        ("diabetes_label", "Diabetes"),
        ("alcoholism_label", "Alcoholism"),
        ("handicap_label", "Disability flag"),
        ("scheduled_hour_bucket", "Scheduling hour"),
        ("neighbourhood", "Neighborhood"),
    ]
)

ORDERED_VALUES = {
    "wait_bucket": [
        "Same day",
        "1 day",
        "2-3 days",
        "4-7 days",
        "8-14 days",
        "15-30 days",
        "31+ days",
    ],
    "age_group": [
        "0-12 child",
        "13-17 teen",
        "18-29 young adult",
        "30-44 adult",
        "45-59 midlife",
        "60-74 senior",
        "75+ older adult",
    ],
    "appointment_weekday": WEEKDAY_ORDER,
    "scheduled_weekday": WEEKDAY_ORDER,
    "scheduled_hour_bucket": ["Early", "Morning", "Midday", "Afternoon", "Evening"],
    "condition_count_bucket": ["0 conditions", "1 condition", "2 conditions", "3+ conditions"],
}


def pct(value: float, decimals: int = 1) -> str:
    return f"{value * 100:.{decimals}f}%"


def pp(value: float, decimals: int = 1) -> str:
    return f"{value * 100:+.{decimals}f} pp"


def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def load_and_prepare() -> tuple[pd.DataFrame, dict[str, int]]:
    df = pd.read_csv(DATA_FILE)
    original_rows = len(df)

    df = df.rename(
        columns={
            "PatientId": "patient_id",
            "AppointmentID": "appointment_id",
            "Gender": "gender",
            "ScheduledDay": "scheduled_day",
            "AppointmentDay": "appointment_day",
            "Age": "age",
            "Neighbourhood": "neighbourhood",
            "Scholarship": "scholarship",
            "Hipertension": "hypertension",
            "Diabetes": "diabetes",
            "Alcoholism": "alcoholism",
            "Handcap": "handicap",
            "SMS_received": "sms_received",
            "No-show": "no_show_raw",
        }
    )
    df["scheduled_day"] = pd.to_datetime(df["scheduled_day"], utc=True)
    df["appointment_day"] = pd.to_datetime(df["appointment_day"], utc=True)
    df["appointment_date"] = df["appointment_day"].dt.normalize()
    df["scheduled_date"] = df["scheduled_day"].dt.normalize()
    df["wait_days"] = (df["appointment_date"] - df["scheduled_date"]).dt.days
    df["no_show"] = df["no_show_raw"].eq("Yes").astype(int)

    invalid_age = int(df["age"].lt(0).sum())
    invalid_wait = int(df["wait_days"].lt(0).sum())
    df = df.loc[df["age"].ge(0) & df["wait_days"].ge(0)].copy()

    df["wait_bucket"] = pd.cut(
        df["wait_days"],
        bins=[-1, 0, 1, 3, 7, 14, 30, np.inf],
        labels=ORDERED_VALUES["wait_bucket"],
    ).astype(str)
    df["age_group"] = pd.cut(
        df["age"],
        bins=[-1, 12, 17, 29, 44, 59, 74, np.inf],
        labels=ORDERED_VALUES["age_group"],
    ).astype(str)
    df["appointment_weekday"] = df["appointment_day"].dt.day_name()
    df["scheduled_weekday"] = df["scheduled_day"].dt.day_name()
    df["scheduled_hour"] = df["scheduled_day"].dt.hour
    df["scheduled_hour_bucket"] = pd.cut(
        df["scheduled_hour"],
        bins=[-1, 6, 10, 14, 17, 23],
        labels=ORDERED_VALUES["scheduled_hour_bucket"],
    ).astype(str)
    df["appointment_month"] = df["appointment_day"].dt.strftime("%Y-%m")
    df["condition_count"] = (
        df["hypertension"].clip(0, 1)
        + df["diabetes"].clip(0, 1)
        + df["alcoholism"].clip(0, 1)
        + df["handicap"].gt(0).astype(int)
    )
    df["condition_count_bucket"] = pd.cut(
        df["condition_count"],
        bins=[-1, 0, 1, 2, 4],
        labels=ORDERED_VALUES["condition_count_bucket"],
    ).astype(str)

    df["scholarship_label"] = np.where(df["scholarship"].eq(1), "Scholarship: yes", "Scholarship: no")
    df["sms_received_label"] = np.where(df["sms_received"].eq(1), "SMS received", "No SMS")
    df["hypertension_label"] = np.where(df["hypertension"].eq(1), "Hypertension", "No hypertension")
    df["diabetes_label"] = np.where(df["diabetes"].eq(1), "Diabetes", "No diabetes")
    df["alcoholism_label"] = np.where(df["alcoholism"].eq(1), "Alcoholism", "No alcoholism")
    df["handicap_label"] = np.where(df["handicap"].gt(0), "Disability flag", "No disability flag")

    removed_rows = original_rows - len(df)
    data_quality = {
        "original_rows": original_rows,
        "clean_rows": len(df),
        "invalid_age_rows": invalid_age,
        "invalid_wait_rows": invalid_wait,
        "removed_rows": removed_rows,
    }
    return df, data_quality


def sort_categories(summary: pd.DataFrame, factor_key: str) -> pd.DataFrame:
    ordered = ORDERED_VALUES.get(factor_key)
    if ordered:
        order_map = {value: i for i, value in enumerate(ordered)}
        return summary.assign(order_key=summary["category"].map(order_map).fillna(999)).sort_values("order_key")
    if factor_key == "neighbourhood":
        return summary.sort_values(["appointments", "no_show_rate"], ascending=[False, False])
    return summary.sort_values("no_show_rate", ascending=False)


def factor_summary(df: pd.DataFrame, factor_key: str, label: str, overall_rate: float) -> pd.DataFrame:
    grouped = (
        df.groupby(factor_key, dropna=False, observed=True)
        .agg(
            appointments=("appointment_id", "size"),
            no_shows=("no_show", "sum"),
            no_show_rate=("no_show", "mean"),
            predicted_no_show_rate=("predicted_no_show_probability", "mean"),
            median_wait_days=("wait_days", "median"),
            avg_age=("age", "mean"),
        )
        .reset_index()
        .rename(columns={factor_key: "category"})
    )
    grouped["factor_key"] = factor_key
    grouped["factor"] = label
    grouped["show_rate"] = 1 - grouped["no_show_rate"]
    grouped["risk_lift"] = grouped["no_show_rate"] / overall_rate - 1
    grouped["difference_pp"] = grouped["no_show_rate"] - overall_rate
    grouped["expected_no_shows_at_average"] = grouped["appointments"] * overall_rate
    grouped["excess_no_shows"] = grouped["no_shows"] - grouped["expected_no_shows_at_average"]
    grouped["category"] = grouped["category"].astype(str)
    grouped = sort_categories(grouped, factor_key)
    grouped["display_rank"] = np.arange(1, len(grouped) + 1)
    return grouped


def fit_prediction_model(df: pd.DataFrame) -> tuple[Pipeline, dict[str, float], pd.DataFrame, pd.DataFrame]:
    categorical_features = [
        "gender",
        "age_group",
        "wait_bucket",
        "appointment_weekday",
        "scheduled_weekday",
        "neighbourhood",
        "scholarship_label",
        "sms_received_label",
        "hypertension_label",
        "diabetes_label",
        "alcoholism_label",
        "handicap_label",
    ]
    numeric_features = ["scheduled_hour", "condition_count"]

    X = df[categorical_features + numeric_features]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", make_ohe(), categorical_features),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1200, solver="lbfgs")),
        ]
    )
    model.fit(X_train, y_train)

    test_probabilities = model.predict_proba(X_test)[:, 1]
    full_probabilities = model.predict_proba(X)[:, 1]

    risk_frame = pd.DataFrame({"actual": y, "predicted_probability": full_probabilities})
    risk_frame["risk_decile"] = pd.qcut(
        risk_frame["predicted_probability"].rank(method="first"),
        q=10,
        labels=[f"D{i}" for i in range(1, 11)],
    )
    deciles = (
        risk_frame.groupby("risk_decile", observed=True)
        .agg(
            appointments=("actual", "size"),
            no_show_rate=("actual", "mean"),
            predicted_no_show_rate=("predicted_probability", "mean"),
        )
        .reset_index()
    )
    deciles["risk_decile"] = deciles["risk_decile"].astype(str)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, test_probabilities)),
        "average_precision": float(average_precision_score(y_test, test_probabilities)),
        "brier_score": float(brier_score_loss(y_test, test_probabilities)),
        "base_rate": float(y_test.mean()),
        "top_decile_actual_rate": float(deciles.loc[deciles["risk_decile"].eq("D10"), "no_show_rate"].iloc[0]),
        "bottom_decile_actual_rate": float(deciles.loc[deciles["risk_decile"].eq("D1"), "no_show_rate"].iloc[0]),
    }

    cat_encoder = model.named_steps["preprocessor"].named_transformers_["cat"]
    cat_names = cat_encoder.get_feature_names_out(categorical_features)
    feature_names = np.r_[numeric_features, cat_names]
    coefs = model.named_steps["model"].coef_[0]
    coefficients = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefs,
            "odds_ratio": np.exp(coefs),
            "direction": np.where(coefs >= 0, "Raises predicted risk", "Lowers predicted risk"),
        }
    )
    coefficients["abs_coefficient"] = coefficients["coefficient"].abs()
    coefficients = coefficients.sort_values("abs_coefficient", ascending=False)
    return model, metrics, coefficients, deciles


def make_month_summary(df: pd.DataFrame, overall_rate: float) -> pd.DataFrame:
    summary = (
        df.groupby("appointment_month", observed=True)
        .agg(
            appointments=("appointment_id", "size"),
            no_shows=("no_show", "sum"),
            no_show_rate=("no_show", "mean"),
            predicted_no_show_rate=("predicted_no_show_probability", "mean"),
            median_wait_days=("wait_days", "median"),
        )
        .reset_index()
    )
    summary["difference_pp"] = summary["no_show_rate"] - overall_rate
    return summary


def make_recommendations(df: pd.DataFrame, factor_data: dict[str, pd.DataFrame], metrics: dict[str, float]) -> dict:
    overall = df["no_show"].mean()
    wait_summary = factor_data["wait_bucket"]
    age_summary = factor_data["age_group"]
    sms_summary = factor_data["sms_received_label"]
    scholarship_summary = factor_data["scholarship_label"]
    neighborhoods = factor_data["neighbourhood"]

    high_wait = wait_summary.loc[wait_summary["category"].eq("31+ days")].iloc[0]
    same_day = wait_summary.loc[wait_summary["category"].eq("Same day")].iloc[0]
    top_age = age_summary.sort_values("no_show_rate", ascending=False).iloc[0]
    top_neighborhood = neighborhoods.loc[neighborhoods["appointments"].ge(500)].sort_values(
        "no_show_rate", ascending=False
    ).iloc[0]
    sms_yes = sms_summary.loc[sms_summary["category"].eq("SMS received")].iloc[0]
    sms_no = sms_summary.loc[sms_summary["category"].eq("No SMS")].iloc[0]
    scholarship_yes = scholarship_summary.loc[scholarship_summary["category"].eq("Scholarship: yes")].iloc[0]
    scholarship_no = scholarship_summary.loc[scholarship_summary["category"].eq("Scholarship: no")].iloc[0]

    actions = [
        {
            "priority": "1",
            "area": "Long lead times",
            "evidence": (
                f"Same-day appointments missed at {pct(same_day['no_show_rate'])}, while 31+ day waits "
                f"missed at {pct(high_wait['no_show_rate'])}."
            ),
            "institute_action": (
                "Protect near-term access, fill cancellations from a waitlist, and require active confirmation "
                "for appointments booked more than a week out."
            ),
        },
        {
            "priority": "2",
            "area": "Reminder design",
            "evidence": (
                f"SMS-received appointments show {pct(sms_yes['no_show_rate'])} no-show versus "
                f"{pct(sms_no['no_show_rate'])} without SMS; this is likely confounded by longer waits."
            ),
            "institute_action": (
                "Do not interpret this as SMS causing no-shows. Use timed, two-way reminders: booking confirmation, "
                "72-hour confirmation, day-before reminder, and easy reschedule links."
            ),
        },
        {
            "priority": "3",
            "area": "Young and socially vulnerable groups",
            "evidence": (
                f"The highest age band is {top_age['category']} at {pct(top_age['no_show_rate'])}; "
                f"scholarship patients missed at {pct(scholarship_yes['no_show_rate'])} versus "
                f"{pct(scholarship_no['no_show_rate'])} for non-scholarship patients."
            ),
            "institute_action": (
                "Add friction-reducing support: flexible slots, transport guidance, caregiver contact options, "
                "and low-penalty rescheduling instead of punitive cancellation policies."
            ),
        },
        {
            "priority": "4",
            "area": "Neighborhood operations",
            "evidence": (
                f"Among neighborhoods with at least 500 appointments, {top_neighborhood['category']} has the "
                f"highest observed no-show rate at {pct(top_neighborhood['no_show_rate'])}."
            ),
            "institute_action": (
                "Target outreach geographically: adjust reminder channels, partner with local primary-care teams, "
                "and consider transport or telehealth alternatives in persistent hot spots."
            ),
        },
        {
            "priority": "5",
            "area": "Risk scoring",
            "evidence": (
                f"The interpretable logistic model reaches ROC AUC {metrics['roc_auc']:.3f}; its highest-risk "
                f"decile misses at {pct(metrics['top_decile_actual_rate'])}."
            ),
            "institute_action": (
                "Use risk tiers for operational triage, not denial of care: extra confirmations, standby lists, "
                "and compassionate rescheduling for high-risk appointments."
            ),
        },
    ]
    return {
        "overall_rate": overall,
        "actions": actions,
        "summary": {
            "same_day_rate": float(same_day["no_show_rate"]),
            "long_wait_rate": float(high_wait["no_show_rate"]),
            "top_age_group": str(top_age["category"]),
            "top_age_rate": float(top_age["no_show_rate"]),
            "top_neighborhood": str(top_neighborhood["category"]),
            "top_neighborhood_rate": float(top_neighborhood["no_show_rate"]),
        },
    }


def write_outputs(
    df: pd.DataFrame,
    data_quality: dict[str, int],
    model_metrics: dict[str, float],
    coefficients: pd.DataFrame,
    deciles: pd.DataFrame,
    factor_data: dict[str, pd.DataFrame],
    month_summary: pd.DataFrame,
    recommendations: dict,
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    pd.concat(factor_data.values(), ignore_index=True).to_csv(OUTPUT_DIR / "factor_no_show_rates.csv", index=False)
    coefficients.to_csv(OUTPUT_DIR / "model_coefficients.csv", index=False)
    deciles.to_csv(OUTPUT_DIR / "risk_deciles.csv", index=False)
    month_summary.to_csv(OUTPUT_DIR / "monthly_no_show_rates.csv", index=False)
    with (OUTPUT_DIR / "recommendations.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "data_quality": data_quality,
                "model_metrics": model_metrics,
                "recommendations": recommendations,
            },
            f,
            indent=2,
        )


def records_for_json(frame: pd.DataFrame) -> list[dict]:
    clean = frame.replace([np.inf, -np.inf], np.nan).copy()
    clean = clean.where(pd.notnull(clean), None)
    return clean.to_dict(orient="records")


def make_dashboard(
    df: pd.DataFrame,
    data_quality: dict[str, int],
    model_metrics: dict[str, float],
    coefficients: pd.DataFrame,
    deciles: pd.DataFrame,
    factor_data: dict[str, pd.DataFrame],
    month_summary: pd.DataFrame,
    recommendations: dict,
) -> str:
    overall_rate = df["no_show"].mean()
    kpis = [
        {"label": "Clean appointments", "value": f"{len(df):,}", "detail": f"{data_quality['removed_rows']} invalid rows removed"},
        {"label": "Overall no-show", "value": pct(overall_rate), "detail": f"{int(df['no_show'].sum()):,} missed visits"},
        {"label": "Median lead time", "value": f"{df['wait_days'].median():.0f} days", "detail": f"Mean {df['wait_days'].mean():.1f} days"},
        {"label": "Risk model AUC", "value": f"{model_metrics['roc_auc']:.3f}", "detail": f"Top decile {pct(model_metrics['top_decile_actual_rate'])}"},
    ]

    strongest_positive = (
        coefficients.loc[~coefficients["feature"].str.startswith("neighbourhood_")]
        .sort_values("coefficient", ascending=False)
        .head(12)
    )
    strongest_negative = (
        coefficients.loc[~coefficients["feature"].str.startswith("neighbourhood_")]
        .sort_values("coefficient")
        .head(12)
    )
    driver_data = pd.concat([strongest_positive, strongest_negative], ignore_index=True)

    payload = {
        "factors": {key: records_for_json(value) for key, value in factor_data.items()},
        "factorLabels": FACTORS,
        "monthSummary": records_for_json(month_summary),
        "deciles": records_for_json(deciles),
        "drivers": records_for_json(driver_data),
        "recommendations": recommendations,
        "kpis": kpis,
        "overallRate": overall_rate,
        "modelMetrics": model_metrics,
    }

    template = Template(
        dedent(
            r"""
            <!doctype html>
            <html lang="en">
            <head>
              <meta charset="utf-8">
              <meta name="viewport" content="width=device-width, initial-scale=1">
              <title>Medical Appointment No-Show Story Dashboard</title>
              <script>{{ plotly_js }}</script>
              <style>
                :root {
                  --bg: #f7f8fb;
                  --panel: #ffffff;
                  --ink: #18212f;
                  --muted: #617083;
                  --line: #dce2ea;
                  --teal: #108b8b;
                  --blue: #3568d4;
                  --rose: #c84b5f;
                  --amber: #c77a18;
                  --green: #2d8a50;
                  --shadow: 0 12px 28px rgba(28, 39, 55, 0.08);
                }
                * { box-sizing: border-box; }
                body {
                  margin: 0;
                  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                  color: var(--ink);
                  background: var(--bg);
                }
                .shell {
                  min-height: 100vh;
                  display: grid;
                  grid-template-columns: 320px minmax(0, 1fr);
                }
                aside {
                  background: #111927;
                  color: #f8fbff;
                  padding: 24px 22px;
                  position: sticky;
                  top: 0;
                  height: 100vh;
                  overflow-y: auto;
                }
                main { padding: 24px; min-width: 0; }
                h1 {
                  font-size: 1.45rem;
                  line-height: 1.15;
                  margin: 0 0 10px;
                  letter-spacing: 0;
                }
                h2 {
                  font-size: 1rem;
                  margin: 0 0 12px;
                  letter-spacing: 0;
                }
                p { color: var(--muted); line-height: 1.5; }
                aside p { color: #b8c3d2; margin: 0 0 22px; }
                label {
                  display: block;
                  font-size: 0.76rem;
                  text-transform: uppercase;
                  letter-spacing: 0.08em;
                  color: #9fb0c4;
                  margin: 18px 0 8px;
                  font-weight: 700;
                }
                select, input[type="range"] {
                  width: 100%;
                }
                select {
                  min-height: 42px;
                  border: 1px solid #324258;
                  border-radius: 6px;
                  padding: 0 12px;
                  color: #f8fbff;
                  background: #182235;
                  font-size: 0.95rem;
                }
                input[type="range"] { accent-color: var(--teal); }
                .range-readout {
                  display: flex;
                  justify-content: space-between;
                  color: #d5deea;
                  font-size: 0.88rem;
                  margin-top: 6px;
                }
                .tabs {
                  display: flex;
                  gap: 8px;
                  flex-wrap: wrap;
                  margin: 0 0 18px;
                }
                .tab {
                  border: 1px solid var(--line);
                  background: #fff;
                  color: var(--ink);
                  border-radius: 6px;
                  padding: 9px 13px;
                  font-weight: 700;
                  cursor: pointer;
                }
                .tab.active {
                  background: #13243d;
                  color: #fff;
                  border-color: #13243d;
                }
                .view { display: none; }
                .view.active { display: block; }
                .kpi-grid {
                  display: grid;
                  grid-template-columns: repeat(4, minmax(0, 1fr));
                  gap: 14px;
                  margin-bottom: 18px;
                }
                .kpi, .panel, .action, .table-wrap {
                  background: var(--panel);
                  border: 1px solid var(--line);
                  border-radius: 8px;
                  box-shadow: var(--shadow);
                }
                .kpi { padding: 16px; min-height: 112px; }
                .kpi .label { color: var(--muted); font-size: 0.82rem; font-weight: 700; }
                .kpi .value { font-size: 2rem; font-weight: 800; margin-top: 8px; letter-spacing: 0; }
                .kpi .detail { color: var(--muted); font-size: 0.88rem; margin-top: 4px; }
                .grid-2 {
                  display: grid;
                  grid-template-columns: minmax(0, 1.35fr) minmax(330px, 0.65fr);
                  gap: 18px;
                }
                .panel { padding: 16px; min-width: 0; }
                .chart { width: 100%; height: 430px; }
                .chart.small { height: 330px; }
                .insight-list {
                  display: grid;
                  gap: 10px;
                }
                .insight {
                  border-left: 4px solid var(--teal);
                  background: #f7fafc;
                  padding: 12px;
                  border-radius: 6px;
                }
                .insight strong { display: block; margin-bottom: 4px; }
                .insight span { color: var(--muted); font-size: 0.92rem; line-height: 1.45; }
                .table-wrap { overflow-x: auto; margin-top: 18px; }
                table { border-collapse: collapse; width: 100%; min-width: 760px; background: #fff; }
                th, td {
                  border-bottom: 1px solid var(--line);
                  padding: 10px 12px;
                  text-align: left;
                  font-size: 0.9rem;
                }
                th {
                  color: var(--muted);
                  font-size: 0.75rem;
                  text-transform: uppercase;
                  letter-spacing: 0.07em;
                  background: #f4f6f9;
                }
                .actions {
                  display: grid;
                  gap: 12px;
                }
                .action { padding: 16px; display: grid; grid-template-columns: 44px minmax(0, 1fr); gap: 12px; }
                .badge {
                  width: 36px;
                  height: 36px;
                  display: grid;
                  place-items: center;
                  border-radius: 50%;
                  background: #e9f7f5;
                  color: var(--teal);
                  font-weight: 800;
                }
                .action h3 { margin: 0 0 6px; font-size: 1rem; }
                .action p { margin: 0 0 8px; }
                .action .do { color: var(--ink); font-weight: 650; }
                .note {
                  background: #fff8eb;
                  border: 1px solid #f0d9ad;
                  border-radius: 8px;
                  padding: 12px 14px;
                  color: #67450f;
                  margin: 12px 0 0;
                  line-height: 1.45;
                }
                @media (max-width: 980px) {
                  .shell { grid-template-columns: 1fr; }
                  aside { position: relative; height: auto; }
                  main { padding: 18px; }
                  .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
                  .grid-2 { grid-template-columns: 1fr; }
                }
                @media (max-width: 560px) {
                  .kpi-grid { grid-template-columns: 1fr; }
                  .chart { height: 360px; }
                }
              </style>
            </head>
            <body>
              <div class="shell">
                <aside>
                  <h1>No-Show Appointment Risk</h1>
                  <p>Interactive story dashboard for the May-June 2016 medical appointment dataset. Rates show missed appointments where <b>No-show = Yes</b>.</p>

                  <label for="factorSelect">Category</label>
                  <select id="factorSelect"></select>

                  <label for="sortSelect">Sort</label>
                  <select id="sortSelect">
                    <option value="default">Natural / volume order</option>
                    <option value="risk">Highest no-show rate</option>
                    <option value="volume">Most appointments</option>
                    <option value="excess">Most excess missed visits</option>
                  </select>

                  <label for="topN">Visible categories</label>
                  <input id="topN" type="range" min="5" max="40" value="18">
                  <div class="range-readout"><span>5</span><b id="topNValue">18</b><span>40</span></div>

                  <label for="minVolume">Minimum appointments</label>
                  <input id="minVolume" type="range" min="1" max="1000" value="30" step="10">
                  <div class="range-readout"><span>1</span><b id="minVolumeValue">30</b><span>1000</span></div>

                  <div class="note">Use the predicted rate as a triage signal, not as a reason to restrict access. The best operational use is extra support for appointments most likely to be missed.</div>
                </aside>

                <main>
                  <div class="tabs">
                    <button class="tab active" data-view="overview">Overview</button>
                    <button class="tab" data-view="explorer">Category Explorer</button>
                    <button class="tab" data-view="model">Prediction</button>
                    <button class="tab" data-view="actions">Actions</button>
                  </div>

                  <section id="overview" class="view active">
                    <div class="kpi-grid" id="kpis"></div>
                    <div class="grid-2">
                      <div class="panel">
                        <h2>No-show rate over appointment month</h2>
                        <div id="monthChart" class="chart"></div>
                      </div>
                      <div class="panel">
                        <h2>Story signals</h2>
                        <div class="insight-list" id="storySignals"></div>
                      </div>
                    </div>
                  </section>

                  <section id="explorer" class="view">
                    <div class="grid-2">
                      <div class="panel">
                        <h2 id="factorTitle">No-show rate by category</h2>
                        <div id="factorChart" class="chart"></div>
                      </div>
                      <div class="panel">
                        <h2>What stands out</h2>
                        <div class="insight-list" id="factorInsights"></div>
                      </div>
                    </div>
                    <div class="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>Category</th>
                            <th>Appointments</th>
                            <th>No-shows</th>
                            <th>Observed rate</th>
                            <th>Predicted rate</th>
                            <th>Lift vs avg</th>
                            <th>Median wait</th>
                          </tr>
                        </thead>
                        <tbody id="factorTable"></tbody>
                      </table>
                    </div>
                  </section>

                  <section id="model" class="view">
                    <div class="grid-2">
                      <div class="panel">
                        <h2>Risk deciles: predicted vs observed</h2>
                        <div id="decileChart" class="chart"></div>
                      </div>
                      <div class="panel">
                        <h2>Strongest non-neighborhood model signals</h2>
                        <div id="driverChart" class="chart"></div>
                      </div>
                    </div>
                  </section>

                  <section id="actions" class="view">
                    <div class="actions" id="actionList"></div>
                  </section>
                </main>
              </div>

              <script id="dashboard-data" type="application/json">{{ payload }}</script>
              <script>
                const DATA = JSON.parse(document.getElementById('dashboard-data').textContent);
                const fmtPct = value => `${(value * 100).toFixed(1)}%`;
                const fmtPp = value => `${value >= 0 ? '+' : ''}${(value * 100).toFixed(1)} pp`;
                const fmtInt = value => Math.round(value).toLocaleString();
                const colors = {
                  observed: '#108b8b',
                  predicted: '#c84b5f',
                  count: '#3568d4',
                  amber: '#c77a18',
                  green: '#2d8a50'
                };
                const baseLayout = {
                  margin: {l: 52, r: 24, t: 18, b: 80},
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(0,0,0,0)',
                  font: {family: 'Inter, system-ui, sans-serif', color: '#18212f'},
                  hoverlabel: {bgcolor: '#111927', bordercolor: '#111927', font: {color: '#fff'}},
                  legend: {orientation: 'h', y: 1.1}
                };

                function populateControls() {
                  const select = document.getElementById('factorSelect');
                  Object.entries(DATA.factorLabels).forEach(([key, label]) => {
                    const option = document.createElement('option');
                    option.value = key;
                    option.textContent = label;
                    select.appendChild(option);
                  });
                  select.value = 'wait_bucket';
                }

                function renderKpis() {
                  const host = document.getElementById('kpis');
                  host.innerHTML = DATA.kpis.map(kpi => `
                    <div class="kpi">
                      <div class="label">${kpi.label}</div>
                      <div class="value">${kpi.value}</div>
                      <div class="detail">${kpi.detail}</div>
                    </div>
                  `).join('');
                }

                function sortedRows() {
                  const factor = document.getElementById('factorSelect').value;
                  const sort = document.getElementById('sortSelect').value;
                  const topN = Number(document.getElementById('topN').value);
                  const minVolume = Number(document.getElementById('minVolume').value);
                  let rows = DATA.factors[factor].filter(row => row.appointments >= minVolume);
                  if (sort === 'risk') rows = rows.sort((a, b) => b.no_show_rate - a.no_show_rate);
                  if (sort === 'volume') rows = rows.sort((a, b) => b.appointments - a.appointments);
                  if (sort === 'excess') rows = rows.sort((a, b) => b.excess_no_shows - a.excess_no_shows);
                  return rows.slice(0, topN);
                }

                function renderFactor() {
                  document.getElementById('topNValue').textContent = document.getElementById('topN').value;
                  document.getElementById('minVolumeValue').textContent = document.getElementById('minVolume').value;
                  const factor = document.getElementById('factorSelect').value;
                  const label = DATA.factorLabels[factor];
                  const rows = sortedRows();
                  document.getElementById('factorTitle').textContent = `No-show rate by ${label.toLowerCase()}`;

                  Plotly.react('factorChart', [
                    {
                      type: 'bar',
                      name: 'Observed no-show',
                      x: rows.map(r => r.category),
                      y: rows.map(r => r.no_show_rate),
                      marker: {color: colors.observed},
                      customdata: rows.map(r => [r.appointments, r.no_shows, r.predicted_no_show_rate, r.risk_lift, r.median_wait_days]),
                      hovertemplate: '<b>%{x}</b><br>Observed: %{y:.1%}<br>Predicted: %{customdata[2]:.1%}<br>Appointments: %{customdata[0]:,}<br>No-shows: %{customdata[1]:,}<br>Lift: %{customdata[3]:+.1%}<br>Median wait: %{customdata[4]:.0f} days<extra></extra>'
                    },
                    {
                      type: 'scatter',
                      mode: 'lines+markers',
                      name: 'Predicted no-show',
                      x: rows.map(r => r.category),
                      y: rows.map(r => r.predicted_no_show_rate),
                      line: {color: colors.predicted, width: 3},
                      marker: {size: 7},
                      hovertemplate: '<b>%{x}</b><br>Predicted: %{y:.1%}<extra></extra>'
                    }
                  ], {
                    ...baseLayout,
                    yaxis: {tickformat: '.0%', rangemode: 'tozero', title: 'No-show rate'},
                    xaxis: {tickangle: -35, automargin: true},
                    shapes: [{
                      type: 'line',
                      xref: 'paper',
                      x0: 0,
                      x1: 1,
                      y0: DATA.overallRate,
                      y1: DATA.overallRate,
                      line: {dash: 'dot', color: '#617083', width: 2}
                    }],
                    annotations: [{
                      xref: 'paper',
                      x: 1,
                      y: DATA.overallRate,
                      text: 'overall',
                      showarrow: false,
                      xanchor: 'left',
                      font: {color: '#617083'}
                    }]
                  }, {responsive: true, displayModeBar: false});

                  renderFactorInsights(rows, label);
                  renderFactorTable(rows);
                }

                function renderFactorInsights(rows, label) {
                  const host = document.getElementById('factorInsights');
                  if (!rows.length) {
                    host.innerHTML = '<div class="insight"><strong>No categories meet the filter.</strong><span>Lower the minimum appointment threshold.</span></div>';
                    return;
                  }
                  const highest = [...rows].sort((a, b) => b.no_show_rate - a.no_show_rate)[0];
                  const excess = [...rows].sort((a, b) => b.excess_no_shows - a.excess_no_shows)[0];
                  const gap = highest.no_show_rate - DATA.overallRate;
                  host.innerHTML = `
                    <div class="insight"><strong>Highest filtered risk</strong><span>${highest.category} has a ${fmtPct(highest.no_show_rate)} observed no-show rate, ${fmtPp(gap)} versus the overall average.</span></div>
                    <div class="insight"><strong>Operational load</strong><span>${excess.category} contributes about ${fmtInt(Math.max(0, excess.excess_no_shows))} excess missed visits versus an average-risk category of the same size.</span></div>
                    <div class="insight"><strong>Prediction check</strong><span>The model-predicted line helps separate persistent risk from noisy small categories; large gaps are worth auditing locally.</span></div>
                  `;
                }

                function renderFactorTable(rows) {
                  const body = document.getElementById('factorTable');
                  body.innerHTML = rows.map(row => `
                    <tr>
                      <td>${row.category}</td>
                      <td>${fmtInt(row.appointments)}</td>
                      <td>${fmtInt(row.no_shows)}</td>
                      <td>${fmtPct(row.no_show_rate)}</td>
                      <td>${fmtPct(row.predicted_no_show_rate)}</td>
                      <td>${fmtPp(row.difference_pp)}</td>
                      <td>${Number(row.median_wait_days).toFixed(0)} days</td>
                    </tr>
                  `).join('');
                }

                function renderOverview() {
                  const monthRows = DATA.monthSummary;
                  Plotly.newPlot('monthChart', [
                    {
                      type: 'bar',
                      name: 'Appointments',
                      x: monthRows.map(r => r.appointment_month),
                      y: monthRows.map(r => r.appointments),
                      yaxis: 'y2',
                      marker: {color: 'rgba(53,104,212,0.25)'},
                      hovertemplate: '<b>%{x}</b><br>Appointments: %{y:,}<extra></extra>'
                    },
                    {
                      type: 'scatter',
                      mode: 'lines+markers',
                      name: 'Observed no-show',
                      x: monthRows.map(r => r.appointment_month),
                      y: monthRows.map(r => r.no_show_rate),
                      line: {color: colors.observed, width: 3},
                      hovertemplate: '<b>%{x}</b><br>No-show: %{y:.1%}<extra></extra>'
                    },
                    {
                      type: 'scatter',
                      mode: 'lines+markers',
                      name: 'Predicted no-show',
                      x: monthRows.map(r => r.appointment_month),
                      y: monthRows.map(r => r.predicted_no_show_rate),
                      line: {color: colors.predicted, width: 3, dash: 'dash'},
                      hovertemplate: '<b>%{x}</b><br>Predicted: %{y:.1%}<extra></extra>'
                    }
                  ], {
                    ...baseLayout,
                    yaxis: {tickformat: '.0%', title: 'No-show rate', rangemode: 'tozero'},
                    yaxis2: {title: 'Appointments', overlaying: 'y', side: 'right', showgrid: false},
                    xaxis: {title: ''}
                  }, {responsive: true, displayModeBar: false});

                  const s = DATA.recommendations.summary;
                  document.getElementById('storySignals').innerHTML = `
                    <div class="insight"><strong>Lead time dominates</strong><span>Same-day visits miss at ${fmtPct(s.same_day_rate)}, while 31+ day waits miss at ${fmtPct(s.long_wait_rate)}.</span></div>
                    <div class="insight"><strong>Age pattern</strong><span>${s.top_age_group} is the highest age band at ${fmtPct(s.top_age_rate)}.</span></div>
                    <div class="insight"><strong>Local hotspot</strong><span>${s.top_neighborhood} is the highest-risk neighborhood with at least 500 appointments, at ${fmtPct(s.top_neighborhood_rate)}.</span></div>
                  `;
                }

                function renderModel() {
                  const deciles = DATA.deciles;
                  Plotly.newPlot('decileChart', [
                    {
                      type: 'bar',
                      name: 'Observed',
                      x: deciles.map(r => r.risk_decile),
                      y: deciles.map(r => r.no_show_rate),
                      marker: {color: colors.observed},
                      hovertemplate: '<b>%{x}</b><br>Observed: %{y:.1%}<extra></extra>'
                    },
                    {
                      type: 'scatter',
                      mode: 'lines+markers',
                      name: 'Predicted',
                      x: deciles.map(r => r.risk_decile),
                      y: deciles.map(r => r.predicted_no_show_rate),
                      line: {color: colors.predicted, width: 3},
                      hovertemplate: '<b>%{x}</b><br>Predicted: %{y:.1%}<extra></extra>'
                    }
                  ], {
                    ...baseLayout,
                    yaxis: {tickformat: '.0%', title: 'No-show rate', rangemode: 'tozero'},
                    xaxis: {title: 'Predicted risk decile'}
                  }, {responsive: true, displayModeBar: false});

                  const drivers = DATA.drivers
                    .map(d => ({...d, cleanFeature: d.feature.replaceAll('_', ' ').replaceAll('  ', ' ')}))
                    .sort((a, b) => a.coefficient - b.coefficient);
                  Plotly.newPlot('driverChart', [{
                    type: 'bar',
                    orientation: 'h',
                    x: drivers.map(d => d.coefficient),
                    y: drivers.map(d => d.cleanFeature),
                    marker: {color: drivers.map(d => d.coefficient >= 0 ? colors.rose : colors.green)},
                    customdata: drivers.map(d => [d.odds_ratio, d.direction]),
                    hovertemplate: '<b>%{y}</b><br>Coefficient: %{x:.2f}<br>Odds ratio: %{customdata[0]:.2f}<br>%{customdata[1]}<extra></extra>'
                  }], {
                    ...baseLayout,
                    margin: {l: 190, r: 24, t: 18, b: 50},
                    xaxis: {title: 'Logistic coefficient'},
                    yaxis: {automargin: true}
                  }, {responsive: true, displayModeBar: false});
                }

                function renderActions() {
                  const host = document.getElementById('actionList');
                  host.innerHTML = DATA.recommendations.actions.map(action => `
                    <article class="action">
                      <div class="badge">${action.priority}</div>
                      <div>
                        <h3>${action.area}</h3>
                        <p>${action.evidence}</p>
                        <p class="do">${action.institute_action}</p>
                      </div>
                    </article>
                  `).join('');
                }

                function setupTabs() {
                  document.querySelectorAll('.tab').forEach(button => {
                    button.addEventListener('click', () => {
                      document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
                      document.querySelectorAll('.view').forEach(view => view.classList.remove('active'));
                      button.classList.add('active');
                      document.getElementById(button.dataset.view).classList.add('active');
                      setTimeout(() => window.dispatchEvent(new Event('resize')), 50);
                    });
                  });
                }

                populateControls();
                renderKpis();
                renderOverview();
                renderFactor();
                renderModel();
                renderActions();
                setupTabs();
                ['factorSelect', 'sortSelect', 'topN', 'minVolume'].forEach(id => {
                  document.getElementById(id).addEventListener('input', renderFactor);
                  document.getElementById(id).addEventListener('change', renderFactor);
                });
              </script>
            </body>
            </html>
            """
        )
    )
    html = template.render(
        plotly_js=get_plotlyjs(),
        payload=json.dumps(payload, ensure_ascii=False).replace("</", "<\\/"),
    )
    DASHBOARD_FILE.write_text(html, encoding="utf-8")
    INDEX_FILE.write_text(html, encoding="utf-8")
    return html


def make_notebook(recommendations: dict) -> None:
    nb = nbf.v4.new_notebook()
    rec_md = "\n".join(
        f"- **{item['area']}**: {item['evidence']} {item['institute_action']}"
        for item in recommendations["actions"]
    )
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            dedent(
                """
                # Medical Appointment No-Show Analysis

                This notebook analyzes the Kaggle medical appointment no-show dataset and focuses on which
                categories predict higher missed-appointment rates. The target is `No-show = Yes`.
                """
            ).strip()
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                from pathlib import Path
                import numpy as np
                import pandas as pd
                import plotly.express as px
                from sklearn.compose import ColumnTransformer
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
                from sklearn.model_selection import train_test_split
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import OneHotEncoder, StandardScaler

                DATA_FILE = Path("noshowappointments-kagglev2-may-2016.csv")
                pd.set_option("display.max_columns", 50)
                """
            ).strip()
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                raw = pd.read_csv(DATA_FILE)
                raw.shape, raw.head()
                """
            ).strip()
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                df = raw.rename(columns={
                    "PatientId": "patient_id",
                    "AppointmentID": "appointment_id",
                    "Gender": "gender",
                    "ScheduledDay": "scheduled_day",
                    "AppointmentDay": "appointment_day",
                    "Age": "age",
                    "Neighbourhood": "neighbourhood",
                    "Scholarship": "scholarship",
                    "Hipertension": "hypertension",
                    "Diabetes": "diabetes",
                    "Alcoholism": "alcoholism",
                    "Handcap": "handicap",
                    "SMS_received": "sms_received",
                    "No-show": "no_show_raw",
                })
                df["scheduled_day"] = pd.to_datetime(df["scheduled_day"], utc=True)
                df["appointment_day"] = pd.to_datetime(df["appointment_day"], utc=True)
                df["appointment_date"] = df["appointment_day"].dt.normalize()
                df["scheduled_date"] = df["scheduled_day"].dt.normalize()
                df["wait_days"] = (df["appointment_date"] - df["scheduled_date"]).dt.days
                df["no_show"] = df["no_show_raw"].eq("Yes").astype(int)

                quality = {
                    "rows_before": len(df),
                    "negative_age_rows": int((df["age"] < 0).sum()),
                    "negative_wait_rows": int((df["wait_days"] < 0).sum()),
                }
                df = df.loc[df["age"].ge(0) & df["wait_days"].ge(0)].copy()
                quality["rows_after"] = len(df)
                quality
                """
            ).strip()
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                wait_order = ["Same day", "1 day", "2-3 days", "4-7 days", "8-14 days", "15-30 days", "31+ days"]
                age_order = ["0-12 child", "13-17 teen", "18-29 young adult", "30-44 adult", "45-59 midlife", "60-74 senior", "75+ older adult"]

                df["wait_bucket"] = pd.cut(df["wait_days"], [-1, 0, 1, 3, 7, 14, 30, np.inf], labels=wait_order).astype(str)
                df["age_group"] = pd.cut(df["age"], [-1, 12, 17, 29, 44, 59, 74, np.inf], labels=age_order).astype(str)
                df["appointment_weekday"] = df["appointment_day"].dt.day_name()
                df["scheduled_weekday"] = df["scheduled_day"].dt.day_name()
                df["scheduled_hour"] = df["scheduled_day"].dt.hour
                df["condition_count"] = (
                    df["hypertension"].clip(0, 1)
                    + df["diabetes"].clip(0, 1)
                    + df["alcoholism"].clip(0, 1)
                    + df["handicap"].gt(0).astype(int)
                )
                df["condition_count_bucket"] = pd.cut(
                    df["condition_count"], [-1, 0, 1, 2, 4],
                    labels=["0 conditions", "1 condition", "2 conditions", "3+ conditions"]
                ).astype(str)
                df["scholarship_label"] = np.where(df["scholarship"].eq(1), "Scholarship: yes", "Scholarship: no")
                df["sms_received_label"] = np.where(df["sms_received"].eq(1), "SMS received", "No SMS")

                overall_rate = df["no_show"].mean()
                overall_rate
                """
            ).strip()
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                def summarize(column):
                    return (
                        df.groupby(column, observed=True)
                        .agg(appointments=("appointment_id", "size"), no_shows=("no_show", "sum"), no_show_rate=("no_show", "mean"), median_wait_days=("wait_days", "median"))
                        .assign(lift_vs_average=lambda x: x["no_show_rate"] / overall_rate - 1)
                        .sort_values("no_show_rate", ascending=False)
                    )

                summarize("wait_bucket").loc[wait_order]
                """
            ).strip()
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                px.bar(
                    summarize("wait_bucket").loc[wait_order].reset_index(),
                    x="wait_bucket",
                    y="no_show_rate",
                    text="appointments",
                    title="No-show rate rises sharply as lead time increases",
                    labels={"no_show_rate": "No-show rate", "wait_bucket": "Lead time"},
                ).update_yaxes(tickformat=".0%")
                """
            ).strip()
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                summarize("age_group").loc[age_order]
                """
            ).strip()
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                key_categories = {
                    "SMS reminder": "sms_received_label",
                    "Scholarship / welfare": "scholarship_label",
                    "Condition count": "condition_count_bucket",
                    "Appointment weekday": "appointment_weekday",
                    "Neighborhood": "neighbourhood",
                }

                category_tables = {}
                for label, column in key_categories.items():
                    table = summarize(column)
                    if column == "neighbourhood":
                        table = table.query("appointments >= 500").head(12)
                    category_tables[label] = table
                category_tables["SMS reminder"], category_tables["Scholarship / welfare"], category_tables["Neighborhood"]
                """
            ).strip()
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                def make_ohe():
                    try:
                        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
                    except TypeError:
                        return OneHotEncoder(handle_unknown="ignore", sparse=True)

                categorical_features = [
                    "gender", "age_group", "wait_bucket", "appointment_weekday", "scheduled_weekday",
                    "neighbourhood", "scholarship_label", "sms_received_label"
                ]
                numeric_features = ["scheduled_hour", "condition_count"]
                X = df[categorical_features + numeric_features]
                y = df["no_show"]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

                model = Pipeline([
                    ("preprocessor", ColumnTransformer([
                        ("num", StandardScaler(), numeric_features),
                        ("cat", make_ohe(), categorical_features),
                    ])),
                    ("model", LogisticRegression(max_iter=1200, solver="lbfgs")),
                ])
                model.fit(X_train, y_train)
                probabilities = model.predict_proba(X_test)[:, 1]

                metrics = {
                    "roc_auc": roc_auc_score(y_test, probabilities),
                    "average_precision": average_precision_score(y_test, probabilities),
                    "brier_score": brier_score_loss(y_test, probabilities),
                    "test_base_rate": y_test.mean(),
                }
                metrics
                """
            ).strip()
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                df["predicted_no_show_probability"] = model.predict_proba(X)[:, 1]
                risk_deciles = pd.DataFrame({"actual": df["no_show"], "predicted": df["predicted_no_show_probability"]})
                risk_deciles["decile"] = pd.qcut(risk_deciles["predicted"].rank(method="first"), 10, labels=[f"D{i}" for i in range(1, 11)])
                decile_summary = risk_deciles.groupby("decile", observed=True).agg(
                    appointments=("actual", "size"),
                    observed_no_show=("actual", "mean"),
                    predicted_no_show=("predicted", "mean"),
                )
                decile_summary
                """
            ).strip()
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                px.line(
                    decile_summary.reset_index(),
                    x="decile",
                    y=["observed_no_show", "predicted_no_show"],
                    markers=True,
                    title="Risk deciles separate lower-risk and higher-risk appointments",
                    labels={"value": "No-show rate", "decile": "Predicted risk decile", "variable": "Series"},
                ).update_yaxes(tickformat=".0%")
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell(
            "## Recommended operational response\n\n"
            + rec_md
            + "\n\nThe model should be used to add support and improve scheduling operations, not to deny care."
        ),
    ]
    nbf.write(nb, NOTEBOOK_FILE)
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute(cwd=BASE_DIR)
    nbf.write(nb, NOTEBOOK_FILE)


def main() -> None:
    df, data_quality = load_and_prepare()
    model, model_metrics, coefficients, deciles = fit_prediction_model(df)

    model_features = [
        "gender",
        "age_group",
        "wait_bucket",
        "appointment_weekday",
        "scheduled_weekday",
        "neighbourhood",
        "scholarship_label",
        "sms_received_label",
        "hypertension_label",
        "diabetes_label",
        "alcoholism_label",
        "handicap_label",
        "scheduled_hour",
        "condition_count",
    ]
    df["predicted_no_show_probability"] = model.predict_proba(df[model_features])[:, 1]
    overall_rate = df["no_show"].mean()
    factor_data = {
        key: factor_summary(df, key, label, overall_rate)
        for key, label in FACTORS.items()
    }
    month_summary = make_month_summary(df, overall_rate)
    recommendations = make_recommendations(df, factor_data, model_metrics)

    write_outputs(
        df,
        data_quality,
        model_metrics,
        coefficients,
        deciles,
        factor_data,
        month_summary,
        recommendations,
    )
    make_dashboard(
        df,
        data_quality,
        model_metrics,
        coefficients,
        deciles,
        factor_data,
        month_summary,
        recommendations,
    )
    make_notebook(recommendations)

    print(f"Wrote {NOTEBOOK_FILE}")
    print(f"Wrote {DASHBOARD_FILE}")
    print(f"Wrote {INDEX_FILE}")
    print(f"Wrote tables under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
