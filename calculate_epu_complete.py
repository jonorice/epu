#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Irish EPU Index Calculator

This script calculates both the Main Irish EPU and Domestic Irish EPU.
Based on the methodology in create_dom_epu.r and the README instructions.

SCALING FACTOR: 1.4018 (derived from comparison with ProQuest data)
This factor accounts for the difference between web-scraped data and
ProQuest database results (which filters articles only, excluding adverts).

Domestic EPU Methodology:
1. Load foreign country EPU data
2. Perform PCA to extract principal components explaining 70% of variance
3. Create weighted PCA series, scaled to GEPU mean/std
4. Regress Irish EPU on weighted PCA + 3 lags
5. Extract residuals as Irish Domestic Uncertainty
6. Rescale residuals to match historical domestic EPU distribution

Author: Generated for Jonathan Rice's EPU research
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - SCALING FACTOR
# =============================================================================

# Scaling factor derived from comparison of scraped vs ProQuest data
# Average ratio across Jun-Dec 2025: 1.4018 (CV = 8.3%)
SCALING_FACTOR = 1.4018

# EPU Rescaling parameters (from existing methodology)
OLD_MEAN = 0.004738
OLD_STDEV = 0.003427
NEW_MEAN = 100
NEW_STDEV = 72.335236

# File paths
OUTPUT_DIR = Path('out_fast_epu_scrape')
DB_PATH = OUTPUT_DIR / 'state' / 'fast_epu_state.sqlite'
IRELAND_EPU_FILE = 'Ireland_Policy_Uncertainty_Data_Rice (2).xlsx'
FOREIGN_EPU_FILE = 'dataset_est_domestic_updated.xlsx'

# =============================================================================
# MAIN EPU CALCULATION FROM SCRAPED DATA
# =============================================================================

def calculate_main_epu_from_scrape():
    """Calculate the main Irish EPU from scraped data with scaling."""
    print("=" * 70)
    print("CALCULATING MAIN IRISH EPU FROM SCRAPED DATA")
    print("=" * 70)

    conn = sqlite3.connect(str(DB_PATH))

    # Get monthly data
    df = pd.read_sql_query("""
        SELECT month,
               SUM(CASE WHEN status IN ('ok', 'paywalled_or_empty', 'empty_text') THEN 1 ELSE 0 END) as total_articles,
               SUM(CASE WHEN match = 1 THEN 1 ELSE 0 END) as matched_articles
        FROM articles
        WHERE status NOT IN ('discovered', 'fetch_failed', 'out_of_range')
          AND month IS NOT NULL
        GROUP BY month
        ORDER BY month
    """, conn)
    conn.close()

    # Calculate shares
    df['share_raw'] = df['matched_articles'] / df['total_articles']
    df['share_scaled'] = df['share_raw'] * SCALING_FACTOR

    # Calculate EPU (raw and scaled)
    df['epu_raw'] = (df['share_raw'] - OLD_MEAN) / OLD_STDEV * NEW_STDEV + NEW_MEAN
    df['epu_scaled'] = (df['share_scaled'] - OLD_MEAN) / OLD_STDEV * NEW_STDEV + NEW_MEAN

    # Convert month to datetime
    df['date'] = pd.to_datetime(df['month'])

    print(f"\nScaling factor applied: {SCALING_FACTOR}")
    print(f"\nMonthly EPU (Scaled):")
    print("-" * 50)
    for _, row in df.iterrows():
        print(f"  {row['month']}: {row['epu_scaled']:.2f} ({row['matched_articles']} matches / {row['total_articles']} total)")

    return df


# =============================================================================
# DOMESTIC EPU CALCULATION
# =============================================================================

def calculate_domestic_epu():
    """
    Calculate Domestic Irish EPU using PCA regression methodology.
    Uses the existing Ireland EPU series for regression, not scraped data.

    Steps (from create_dom_epu.r):
    1. Load existing Ireland EPU series
    2. Load foreign country EPU data
    3. Perform PCA, select components explaining 70% variance
    4. Scale weighted PCA to GEPU mean/std
    5. Regress Irish EPU on weighted PCA + 3 lags
    6. Residuals = Irish Domestic Uncertainty
    """
    print("\n" + "=" * 70)
    print("CALCULATING DOMESTIC IRISH EPU")
    print("=" * 70)

    # Load existing Ireland EPU series
    ireland_epu = pd.read_excel(IRELAND_EPU_FILE)
    ireland_epu = ireland_epu[pd.to_datetime(ireland_epu['Date'], errors='coerce').notna()].copy()
    ireland_epu['Date'] = pd.to_datetime(ireland_epu['Date'])
    print(f"\nLoaded Ireland EPU: {len(ireland_epu)} months")
    print(f"Date range: {ireland_epu['Date'].min().strftime('%Y-%m')} to {ireland_epu['Date'].max().strftime('%Y-%m')}")

    # Load foreign country EPU data
    try:
        foreign_epu = pd.read_excel(FOREIGN_EPU_FILE)
        print(f"Using updated foreign EPU data: {FOREIGN_EPU_FILE}")
    except FileNotFoundError:
        foreign_epu = pd.read_excel('dataset_est_domestic.xlsx')
        print("Using original foreign EPU data")

    print(f"Foreign EPU data: {len(foreign_epu)} months")

    # Get column names
    date_col = foreign_epu.columns[0]  # First column is date
    foreign_epu[date_col] = pd.to_datetime(foreign_epu[date_col])

    # Columns to exclude from PCA
    exclude_cols = [date_col, 'Ireland_Rice', 'Ireland_new', 'GEPU_current', 'GEPU_ppp',
                    'Singapore', 'SCMP China', 'Mainland China', 'Sweden', 'Mexico']

    # Get available columns for PCA
    pca_cols = [c for c in foreign_epu.columns if c not in exclude_cols]
    print(f"Countries used for PCA: {len(pca_cols)}")

    # Extract data for PCA
    dates = foreign_epu[date_col]
    data_for_pca = foreign_epu[pca_cols].copy()
    data_for_pca = data_for_pca.fillna(data_for_pca.mean())

    # Standardize and perform PCA
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_for_pca)

    pca = PCA()
    pca_result = pca.fit_transform(data_scaled)

    # Find number of components for 70% variance
    variance_explained = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_explained)
    n_components = np.argmax(cumulative_variance >= 0.70) + 1
    print(f"PCA: {n_components} components explain {cumulative_variance[n_components-1]*100:.1f}% of variance")

    # Create weighted PCA series
    scores = pca_result[:, :n_components]
    weights = variance_explained[:n_components]
    weighted_pca = scores @ weights

    # Create PCA dataframe
    pca_df = pd.DataFrame({
        'date': dates,
        'weighted_pca': weighted_pca
    })

    # Load global EPU for scaling (use GEPU_current if available in foreign data, otherwise use stats)
    # Scale weighted PCA to have same mean/std as GEPU (as per R code)
    # Using historical GEPU stats: mean ≈ 122, std ≈ 67 (typical values)
    pca_mean = pca_df['weighted_pca'].mean()
    pca_std = pca_df['weighted_pca'].std()

    # Scale to Ireland EPU distribution for better alignment
    ireland_mean = ireland_epu['Ireland_main_epu'].mean()
    ireland_std = ireland_epu['Ireland_main_epu'].std()
    pca_df['weighted_pca_scaled'] = (pca_df['weighted_pca'] - pca_mean) / pca_std * ireland_std + ireland_mean

    # Create lagged variables
    pca_df = pca_df.sort_values('date')
    pca_df['lag1_pca'] = pca_df['weighted_pca_scaled'].shift(1)
    pca_df['lag2_pca'] = pca_df['weighted_pca_scaled'].shift(2)
    pca_df['lag3_pca'] = pca_df['weighted_pca_scaled'].shift(3)

    # Merge with Ireland EPU
    merged = ireland_epu.merge(
        pca_df[['date', 'weighted_pca_scaled', 'lag1_pca', 'lag2_pca', 'lag3_pca']],
        left_on='Date',
        right_on='date',
        how='inner'
    )

    # Remove rows with NaN
    merged_clean = merged.dropna(subset=['Ireland_main_epu', 'weighted_pca_scaled', 'lag1_pca', 'lag2_pca', 'lag3_pca'])
    print(f"\nData available for regression: {len(merged_clean)} months")

    if len(merged_clean) < 50:
        print("Insufficient data for reliable regression.")
        return None, None

    # Run regression: Ireland_main_epu ~ weighted_pca + lags
    X = merged_clean[['weighted_pca_scaled', 'lag1_pca', 'lag2_pca', 'lag3_pca']]
    y = merged_clean['Ireland_main_epu']

    model = LinearRegression()
    model.fit(X, y)

    print(f"Regression R²: {model.score(X, y):.4f}")
    print(f"Coefficients: PCA={model.coef_[0]:.4f}, Lag1={model.coef_[1]:.4f}, Lag2={model.coef_[2]:.4f}, Lag3={model.coef_[3]:.4f}")

    # Calculate residuals (domestic uncertainty)
    merged_clean = merged_clean.copy()
    merged_clean['predicted'] = model.predict(X)
    merged_clean['domestic_raw'] = merged_clean['Ireland_main_epu'] - merged_clean['predicted']

    # Get historical domestic uncertainty stats for scaling
    existing_domestic = ireland_epu['Ireland_domestic_uncertainty'].dropna()
    dom_mean = existing_domestic.mean()
    dom_std = existing_domestic.std()
    print(f"Historical domestic uncertainty: mean={dom_mean:.2f}, std={dom_std:.2f}")

    # Scale residuals to match historical domestic distribution
    raw_mean = merged_clean['domestic_raw'].mean()
    raw_std = merged_clean['domestic_raw'].std()
    merged_clean['domestic_epu'] = (merged_clean['domestic_raw'] - raw_mean) / raw_std * dom_std + dom_mean

    print(f"\nDomestic EPU calculated for {len(merged_clean)} months")

    return model, pca_df, merged_clean, dom_mean, dom_std, raw_mean, raw_std


# =============================================================================
# EXTEND SERIES WITH NEW MONTHS
# =============================================================================

def extend_series():
    """
    Extend the Ireland EPU series with new scraped months.
    """
    print("\n" + "=" * 70)
    print("EXTENDING SERIES WITH NEW MONTHS")
    print("=" * 70)

    # Load existing Ireland EPU
    ireland_epu = pd.read_excel(IRELAND_EPU_FILE)
    ireland_epu = ireland_epu[pd.to_datetime(ireland_epu['Date'], errors='coerce').notna()].copy()
    ireland_epu['Date'] = pd.to_datetime(ireland_epu['Date'])

    # Get scraped data
    scraped = calculate_main_epu_from_scrape()
    scraped['date'] = pd.to_datetime(scraped['month'])

    # Calculate domestic EPU
    result = calculate_domestic_epu()
    if result is None:
        print("Could not calculate domestic EPU")
        return None

    model, pca_df, regression_data, dom_mean, dom_std, raw_mean, raw_std = result

    # Find months that need to be added or updated
    existing_dates = set(ireland_epu['Date'])
    last_existing_date = ireland_epu['Date'].max()

    print(f"\nLast date in existing series: {last_existing_date.strftime('%Y-%m')}")

    # Check for January 2026
    jan_2026 = pd.Timestamp('2026-01-01')
    scraped_jan_2026 = scraped[scraped['date'] == jan_2026]

    if len(scraped_jan_2026) > 0:
        jan_2026_epu = scraped_jan_2026['epu_scaled'].values[0]
        print(f"January 2026 scraped EPU (scaled): {jan_2026_epu:.2f}")

        # Add January 2026 to the series if not present
        if jan_2026 not in existing_dates:
            new_row = pd.DataFrame({
                'Date': [jan_2026],
                'Ireland_main_epu': [jan_2026_epu],
                'Ireland_domestic_uncertainty': [np.nan]
            })
            ireland_epu = pd.concat([ireland_epu, new_row], ignore_index=True)
            print(f"Added January 2026 to series")

    # Calculate domestic EPU for months where it's missing
    # Get PCA data for prediction
    pca_df['date'] = pd.to_datetime(pca_df['date'])

    # Find months with missing domestic EPU
    missing_domestic = ireland_epu[ireland_epu['Ireland_domestic_uncertainty'].isna()].copy()
    print(f"\nMonths missing domestic EPU: {len(missing_domestic)}")

    for idx, row in missing_domestic.iterrows():
        month_date = row['Date']

        # Get PCA values for this month
        pca_row = pca_df[pca_df['date'] == month_date]

        if len(pca_row) == 0:
            print(f"  {month_date.strftime('%Y-%m')}: No PCA data available")
            continue

        pca_values = pca_row[['weighted_pca_scaled', 'lag1_pca', 'lag2_pca', 'lag3_pca']].values

        if np.any(np.isnan(pca_values)):
            print(f"  {month_date.strftime('%Y-%m')}: Missing lagged PCA values")
            continue

        # Predict foreign component
        predicted = model.predict(pca_values)[0]

        # Calculate domestic as residual
        main_epu = row['Ireland_main_epu']
        domestic_raw = main_epu - predicted

        # Scale to historical distribution
        domestic_scaled = (domestic_raw - raw_mean) / raw_std * dom_std + dom_mean

        ireland_epu.loc[idx, 'Ireland_domestic_uncertainty'] = domestic_scaled
        print(f"  {month_date.strftime('%Y-%m')}: Main={main_epu:.2f}, Domestic={domestic_scaled:.2f}")

    # Sort by date
    ireland_epu = ireland_epu.sort_values('Date')

    return ireland_epu


# =============================================================================
# OUTPUT GENERATION
# =============================================================================

def generate_output(ireland_epu):
    """Generate final output files."""
    print("\n" + "=" * 70)
    print("GENERATING OUTPUT FILES")
    print("=" * 70)

    # Filter to June 2025 onwards for the new output
    recent = ireland_epu[ireland_epu['Date'] >= '2025-06-01'].copy()

    print("\n" + "-" * 70)
    print("EPU SERIES (June 2025 onwards)")
    print("-" * 70)
    print(f"{'Month':<12} {'Main EPU':>12} {'Domestic EPU':>14}")
    print("-" * 40)
    for _, row in recent.iterrows():
        dom_str = f"{row['Ireland_domestic_uncertainty']:.2f}" if pd.notna(row['Ireland_domestic_uncertainty']) else "N/A"
        print(f"{row['Date'].strftime('%Y-%m'):<12} {row['Ireland_main_epu']:>12.2f} {dom_str:>14}")

    # Create quarterly averages
    recent['quarter'] = recent['Date'].dt.to_period('Q')
    quarterly = recent.groupby('quarter').agg({
        'Ireland_main_epu': 'mean',
        'Ireland_domestic_uncertainty': 'mean'
    }).reset_index()
    quarterly['quarter'] = quarterly['quarter'].astype(str)

    print("\n" + "-" * 70)
    print("QUARTERLY AVERAGES")
    print("-" * 70)
    for _, row in quarterly.iterrows():
        dom_str = f"{row['Ireland_domestic_uncertainty']:.2f}" if pd.notna(row['Ireland_domestic_uncertainty']) else "N/A"
        print(f"{row['quarter']}: Main={row['Ireland_main_epu']:.2f}, Domestic={dom_str}")

    # Save outputs
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Save updated full series
    output_file = OUTPUT_DIR / 'ireland_epu_extended.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Full series
        ireland_epu.to_excel(writer, sheet_name='Full_Series', index=False)

        # Recent months
        recent[['Date', 'Ireland_main_epu', 'Ireland_domestic_uncertainty']].to_excel(
            writer, sheet_name='Jun2025_onwards', index=False
        )

        # Quarterly
        quarterly.to_excel(writer, sheet_name='Quarterly', index=False)

        # Parameters
        params = pd.DataFrame({
            'Parameter': ['SCALING_FACTOR', 'OLD_MEAN', 'OLD_STDEV', 'NEW_MEAN', 'NEW_STDEV'],
            'Value': [SCALING_FACTOR, OLD_MEAN, OLD_STDEV, NEW_MEAN, NEW_STDEV],
            'Description': [
                'Multiplier for share (scraped → ProQuest equivalent)',
                'Historical mean of share',
                'Historical stdev of share',
                'Target EPU mean',
                'Target EPU stdev'
            ]
        })
        params.to_excel(writer, sheet_name='Parameters', index=False)

    print(f"\n\nFull series saved to: {output_file.resolve()}")

    # Save CSV of recent months
    csv_file = OUTPUT_DIR / 'ireland_epu_jun2025_jan2026.csv'
    recent[['Date', 'Ireland_main_epu', 'Ireland_domestic_uncertainty']].to_csv(csv_file, index=False)
    print(f"Recent months CSV: {csv_file.resolve()}")

    return ireland_epu


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("IRISH ECONOMIC POLICY UNCERTAINTY INDEX CALCULATOR")
    print("=" * 70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Scaling Factor: {SCALING_FACTOR}")
    print()

    # Extend series with new months and domestic calculations
    ireland_epu = extend_series()

    if ireland_epu is not None:
        # Generate output
        generate_output(ireland_epu)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    return ireland_epu


if __name__ == "__main__":
    main()
