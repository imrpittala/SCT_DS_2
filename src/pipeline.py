"""
Titanic Exploratory Data Analysis & Data Cleaning Pipeline
==========================================================
Author: Raghavendra (imrpittala)
Project: SCT_DS_2 - SkillCraft Technology Data Science Internship
Objective: Robust missing value imputation, domain-driven feature engineering,
           and bivariate/multivariate survival pattern correlation analysis.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def set_visual_style():
    """Sets publication-quality seaborn and matplotlib theme."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 1.0,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "figure.titlesize": 16,
        "axes.titlesize": 12.5,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def load_data(filepath: str) -> pd.DataFrame:
    """Loads raw Titanic dataset from CSV."""
    logger.info(f"Loading raw Titanic dataset from: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {df.shape[0]} passenger rows with {df.shape[1]} attributes.")
    return df


def audit_missing_values(df: pd.DataFrame) -> dict:
    """Performs a thorough missing value diagnostic audit."""
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df)) * 100
    audit = {}
    for col in df.columns:
        if null_counts[col] > 0:
            audit[col] = {
                "missing_count": int(null_counts[col]),
                "missing_percentage": round(float(null_pct[col]), 2)
            }
    logger.info(f"Missing Value Audit: {audit}")
    return audit


def extract_title(name: str) -> str:
    """Extracts social/honorific title from passenger name."""
    if pd.isna(name):
        return "Unknown"
    title = name.split(",")[1].split(".")[0].strip()
    
    # Consolidate rare titles into standardized buckets
    title_mapping = {
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Miss",
        "Lady": "Royalty",
        "Countess": "Royalty",
        "Dona": "Royalty",
        "Sir": "Royalty",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Major": "Officer",
        "Col": "Officer",
        "Capt": "Officer",
        "Rev": "Officer",
        "Dr": "Officer"
    }
    return title_mapping.get(title, title if title in ["Mr", "Miss", "Mrs", "Master"] else "Rare")


def impute_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values with contextual statistical strategies:
    - Title extracted first to improve Age imputation accuracy.
    - Age: Imputed by median within (Pclass, Title) groups.
    - Embarked: Imputed using mode port of embarkation.
    - Cabin: Deck letter extracted, missing cabins designated as 'U' (Unknown Deck).
    - Fare: Imputed by Pclass median in case of any zero/null entries.
    """
    logger.info("Executing contextual imputation pipeline...")
    df_clean = df.copy()

    # 1. Title extraction needed for grouped Age imputation
    df_clean["Title"] = df_clean["Name"].apply(extract_title)

    # 2. Age imputation grouped by Pclass and Title
    median_ages = df_clean.groupby(["Pclass", "Title"])["Age"].median()
    df_clean["Age"] = df_clean.apply(
        lambda row: median_ages.get((row["Pclass"], row["Title"]), df_clean["Age"].median())
        if pd.isna(row["Age"]) else row["Age"],
        axis=1
    )

    # 3. Embarked imputation using statistical mode
    mode_embarked = df_clean["Embarked"].mode()[0]
    df_clean["Embarked"] = df_clean["Embarked"].fillna(mode_embarked)

    # 4. Cabin Deck extraction with Unknown flag
    df_clean["Deck"] = df_clean["Cabin"].apply(lambda c: str(c)[0] if pd.notna(c) else "U")
    df_clean["Cabin"] = df_clean["Cabin"].fillna("Unknown")

    # 5. Fare validation
    pclass_median_fare = df_clean.groupby("Pclass")["Fare"].median()
    df_clean["Fare"] = df_clean.apply(
        lambda row: pclass_median_fare.get(row["Pclass"], df_clean["Fare"].median())
        if pd.isna(row["Fare"]) or row["Fare"] == 0 else row["Fare"],
        axis=1
    )

    remaining_nulls = df_clean.isnull().sum().sum()
    logger.info(f"Imputation completed. Total remaining null values: {remaining_nulls}")
    return df_clean


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates informative predictive features for survival analysis:
    - FamilySize = SibSp + Parch + 1
    - IsAlone: Binary indicator for solo travelers
    - Fare_Per_Person = Fare / FamilySize
    - Age_Group: Demographic age brackets
    - Fare_Tier: Quantile-based fare brackets
    """
    logger.info("Engineering domain-specific features...")
    df_fe = df.copy()

    df_fe["FamilySize"] = df_fe["SibSp"] + df_fe["Parch"] + 1
    df_fe["IsAlone"] = (df_fe["FamilySize"] == 1).astype(int)
    df_fe["Fare_Per_Person"] = df_fe["Fare"] / df_fe["FamilySize"]

    # Age bins
    age_bins = [0, 12, 18, 35, 60, 120]
    age_labels = ["Child", "Adolescent", "Young Adult", "Middle Aged", "Senior"]
    df_fe["Age_Group"] = pd.cut(df_fe["Age"], bins=age_bins, labels=age_labels, right=False)

    # Fare quartiles
    df_fe["Fare_Tier"] = pd.qcut(df_fe["Fare"], q=4, labels=["Budget", "Standard", "Premium", "Luxury"])

    return df_fe


def generate_survival_visuals(df: pd.DataFrame, output_image_path: str):
    """
    Generates a 6-panel executive survival trend dashboard:
    1. Overall Survival Rate & Class Disparity (Pclass)
    2. Gender Disparity & Lady First Survival Principle (Sex)
    3. Age Distribution by Survival Outcome (KDE)
    4. Survival by Family Size & Solo Status
    5. Survival Rate across Embarkation Ports
    6. Survival Correlation Matrix
    """
    logger.info("Rendering publication-ready survival visualization grid...")
    set_visual_style()
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    plt.subplots_adjust(hspace=0.35, wspace=0.28)

    fig.suptitle(
        "Titanic Survival Pattern & Exploratory Data Analysis Dashboard\n"
        "SkillCraft Technology Data Science Internship | Task 2 (SCT_DS_2)",
        fontsize=18,
        fontweight="bold",
        y=0.99
    )

    # Panel 1: Survival by Ticket Class (Pclass)
    pclass_surv = df.groupby("Pclass")["Survived"].mean().reset_index()
    sns.barplot(
        data=pclass_surv,
        x="Pclass",
        y="Survived",
        palette=["#2b5c8f", "#4f81bd", "#95b3d7"],
        ax=axes[0, 0],
        edgecolor="black"
    )
    for i, row in pclass_surv.iterrows():
        axes[0, 0].text(
            i, row["Survived"] + 0.02,
            f"{row['Survived'] * 100:.1f}%",
            ha="center", fontweight="bold", fontsize=10.5
        )
    axes[0, 0].set_ylim(0, 1.0)
    axes[0, 0].set_title("Survival Rate by Passenger Class (Socioeconomic Stratum)", fontweight="bold")
    axes[0, 0].set_xlabel("Passenger Class (1st = Upper, 3rd = Steerage)")
    axes[0, 0].set_ylabel("Survival Rate")

    # Panel 2: Survival by Gender (Sex) & Class
    sns.barplot(
        data=df,
        x="Sex",
        y="Survived",
        hue="Pclass",
        palette="Blues_r",
        ax=axes[0, 1],
        edgecolor="black",
        errorbar=None
    )
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_title("Survival by Gender & Class ('Women & Children First')", fontweight="bold")
    axes[0, 1].set_xlabel("Gender")
    axes[0, 1].set_ylabel("Survival Rate")
    axes[0, 1].legend(title="Class", frameon=True)

    # Panel 3: Age Distribution by Survival Outcome (Continuous KDE)
    sns.kdeplot(
        data=df[df["Survived"] == 1],
        x="Age",
        label="Survived",
        color="#2e7d32",
        linewidth=2.5,
        fill=True,
        alpha=0.3,
        ax=axes[1, 0]
    )
    sns.kdeplot(
        data=df[df["Survived"] == 0],
        x="Age",
        label="Did Not Survive",
        color="#c62828",
        linewidth=2.5,
        fill=True,
        alpha=0.3,
        ax=axes[1, 0]
    )
    axes[1, 0].set_title("Age Distribution by Survival Outcome (KDE Density)", fontweight="bold")
    axes[1, 0].set_xlabel("Age (Years)")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].legend(frameon=True)

    # Panel 4: Survival Rate by Family Size
    fam_surv = df.groupby("FamilySize")["Survived"].mean().reset_index()
    sns.barplot(
        data=fam_surv,
        x="FamilySize",
        y="Survived",
        color="#3949ab",
        ax=axes[1, 1],
        edgecolor="black"
    )
    for i, row in fam_surv.iterrows():
        axes[1, 1].text(
            i, row["Survived"] + 0.02,
            f"{row['Survived'] * 100:.0f}%",
            ha="center", fontsize=9, fontweight="bold"
        )
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_title("Survival Rate vs Family Size (Solo vs Large Families)", fontweight="bold")
    axes[1, 1].set_xlabel("Family Size (SibSp + Parch + 1)")
    axes[1, 1].set_ylabel("Survival Rate")

    # Panel 5: Survival Rate across Embarkation Ports (C = Cherbourg, Q = Queenstown, S = Southampton)
    emb_surv = df.groupby("Embarked")["Survived"].mean().reset_index()
    port_names = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}
    emb_surv["Port"] = emb_surv["Embarked"].map(port_names)
    sns.barplot(
        data=emb_surv,
        x="Port",
        y="Survived",
        palette="crest",
        ax=axes[2, 0],
        edgecolor="black"
    )
    for i, row in emb_surv.iterrows():
        axes[2, 0].text(
            i, row["Survived"] + 0.02,
            f"{row['Survived'] * 100:.1f}%",
            ha="center", fontweight="bold", fontsize=10.5
        )
    axes[2, 0].set_ylim(0, 1.0)
    axes[2, 0].set_title("Survival Rate by Embarkation Port", fontweight="bold")
    axes[2, 0].set_xlabel("Port of Embarkation")
    axes[2, 0].set_ylabel("Survival Rate")

    # Panel 6: Correlation Heatmap of Numerical & Engineered Features
    num_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone", "Fare_Per_Person"]
    corr_mat = df[num_cols].corr()
    sns.heatmap(
        corr_mat,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        vmin=-1,
        vmax=1,
        ax=axes[2, 1],
        linewidths=0.5
    )
    axes[2, 1].set_title("Feature Correlation Matrix with Survival", fontweight="bold")

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Survival insights dashboard exported to: {output_image_path}")


def export_eda_summary(df: pd.DataFrame, output_json_path: str):
    """Exports quantitative summary of findings to JSON."""
    summary = {
        "total_passengers": int(len(df)),
        "overall_survival_rate_pct": round(float(df["Survived"].mean() * 100), 2),
        "survival_by_gender": {
            "female_survival_pct": round(float(df[df["Sex"] == "female"]["Survived"].mean() * 100), 2),
            "male_survival_pct": round(float(df[df["Sex"] == "male"]["Survived"].mean() * 100), 2)
        },
        "survival_by_class": {
            str(cls): round(float(rate * 100), 2)
            for cls, rate in df.groupby("Pclass")["Survived"].mean().items()
        },
        "survival_by_title": {
            title: round(float(rate * 100), 2)
            for title, rate in df.groupby("Title")["Survived"].mean().items()
        },
        "solo_vs_family_survival": {
            "is_alone": round(float(df[df["IsAlone"] == 1]["Survived"].mean() * 100), 2),
            "with_family": round(float(df[df["IsAlone"] == 0]["Survived"].mean() * 100), 2)
        }
    }
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    logger.info(f"EDA Summary metrics exported to: {output_json_path}")


def main():
    """Main execution function for Titanic EDA and Cleaning pipeline."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, "data", "dataset.csv")
    cleaned_path = os.path.join(base_dir, "data", "cleaned_dataset.csv")
    report_img_path = os.path.join(base_dir, "reports", "visual_insights.png")
    report_json_path = os.path.join(base_dir, "reports", "eda_summary.json")

    logger.info("Starting Titanic EDA & Data Cleaning Pipeline...")
    df_raw = load_data(raw_path)
    audit_missing_values(df_raw)

    # Impute and Engineer
    df_imputed = impute_missing_data(df_raw)
    df_final = engineer_features(df_imputed)

    # Save cleaned dataset
    df_final.to_csv(cleaned_path, index=False)
    logger.info(f"Cleaned dataset saved to: {cleaned_path}")

    # Generate visual report and metrics
    generate_survival_visuals(df_final, report_img_path)
    export_eda_summary(df_final, report_json_path)
    logger.info("SCT_DS_2 pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
