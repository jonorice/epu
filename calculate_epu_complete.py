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
3. Create weighted PCA series
4. Regress Irish EPU on weighted PCA + 3 lags
5. Extract residuals as Irish Domestic Uncertainty

Author: Generated for Jonathan Rice's EPU research
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

# Output directory
OUTPUT_DIR = Path('out_fast_epu_scrape')
DB_PATH = OUTPUT_DIR / 'state' / 'fast_epu_state.sqlite'

# =============================================================================
# MAIN EPU CALCULATION
# =============================================================================

def calculate_main_epu():
    """Calculate the main Irish EPU from scraped data with scaling."""
    print("=" * 70)
    print("CALCULATING MAIN IRISH EPU")
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

def calculate_domestic_epu(main_epu_df):
    """
    Calculate Domestic Irish EPU using PCA regression methodology.

    Steps (from create_dom_epu.r):
    1. Load foreign country EPU data (excluding Ireland, GEPU, etc.)
    2. Perform PCA, select components explaining 70% variance
    3. Create weighted PCA series
    4. Regress Irish EPU on weighted PCA + 3 lags
    5. Residuals = Irish Domestic Uncertainty
    """
    print("\n" + "=" * 70)
    print("CALCULATING DOMESTIC IRISH EPU")
    print("=" * 70)

    # Load the foreign country EPU data
    # Try updated file first (downloaded from policyuncertainty.com), then fall back to original
    try:
        foreign_epu = pd.read_excel('dataset_est_domestic_updated.xlsx')
        print("Using updated foreign EPU data (from policyuncertainty.com)")
    except FileNotFoundError:
        try:
            foreign_epu = pd.read_excel('dataset_est_domestic.xlsx')
            print("Using original foreign EPU data")
        except FileNotFoundError:
            print("WARNING: No foreign EPU data file found.")
            print("Cannot calculate domestic EPU without foreign country data.")
            return None

    print(f"\nLoaded foreign EPU data: {foreign_epu.shape[0]} rows, {foreign_epu.shape[1]} columns")

    # Get column names
    date_col = foreign_epu.columns[0]  # First column is date

    # Columns to exclude (as per README)
    exclude_cols = [date_col, 'Ireland_Rice', 'Ireland_new', 'GEPU_current', 'GEPU_ppp',
                    'Singapore', 'SCMP China', 'Mainland China', 'Sweden', 'Mexico']

    # Get available columns for PCA
    pca_cols = [c for c in foreign_epu.columns if c not in exclude_cols]
    print(f"Countries used for PCA: {len(pca_cols)}")
    print(f"  {', '.join(pca_cols[:5])}...")

    # Extract data for PCA
    dates = foreign_epu[date_col]
    data_for_pca = foreign_epu[pca_cols].copy()

    # Handle missing values
    data_for_pca = data_for_pca.fillna(data_for_pca.mean())

    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_for_pca)

    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(data_scaled)

    # Calculate variance explained
    variance_explained = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_explained)

    # Find number of components for 70% variance (as per R script)
    n_components = np.argmax(cumulative_variance >= 0.70) + 1
    print(f"\nPCA: {n_components} components explain {cumulative_variance[n_components-1]*100:.1f}% of variance")

    # Create weighted PCA series
    scores = pca_result[:, :n_components]
    weights = variance_explained[:n_components]
    weighted_pca = scores @ weights

    # Create dataframe with dates and weighted PCA
    pca_df = pd.DataFrame({
        'date': pd.to_datetime(dates),
        'weighted_pca': weighted_pca
    })

    # Normalize weighted PCA to have similar scale to EPU
    pca_mean = pca_df['weighted_pca'].mean()
    pca_std = pca_df['weighted_pca'].std()

    # Get Ireland EPU from original data if available (for scaling reference)
    ireland_col = 'Ireland_Rice' if 'Ireland_Rice' in foreign_epu.columns else 'Ireland_new' if 'Ireland_new' in foreign_epu.columns else None
    if ireland_col:
        ireland_epu = foreign_epu[ireland_col].values
        gepu_mean = np.nanmean(ireland_epu)
        gepu_std = np.nanstd(ireland_epu)
        pca_df['weighted_pca_scaled'] = (pca_df['weighted_pca'] - pca_mean) / pca_std * gepu_std + gepu_mean
    else:
        pca_df['weighted_pca_scaled'] = (pca_df['weighted_pca'] - pca_mean) / pca_std * NEW_STDEV + NEW_MEAN

    # Now merge with main EPU data
    main_epu_df['date'] = pd.to_datetime(main_epu_df['month'])

    # Create lagged PCA variables using all available historical data
    pca_df = pca_df.sort_values('date')
    pca_df['lag1_pca'] = pca_df['weighted_pca_scaled'].shift(1)
    pca_df['lag2_pca'] = pca_df['weighted_pca_scaled'].shift(2)
    pca_df['lag3_pca'] = pca_df['weighted_pca_scaled'].shift(3)

    # APPROACH: Use official Ireland EPU from policyuncertainty.com to estimate
    # the regression relationship, then apply to newer scraped months

    # Get Ireland EPU from the foreign data
    if ireland_col:
        pca_df['ireland_official'] = foreign_epu[ireland_col].values
    else:
        print("\nNo Ireland EPU column found in foreign data.")
        return None

    # Remove rows with NaN (from lagging and missing data)
    pca_clean = pca_df.dropna(subset=['lag1_pca', 'lag2_pca', 'lag3_pca', 'ireland_official'])

    print(f"\nHistorical data available for regression: {len(pca_clean)} months")
    print(f"Date range: {pca_clean['date'].min()} to {pca_clean['date'].max()}")

    if len(pca_clean) < 50:
        print("Insufficient historical data for reliable regression.")
        return None

    # Estimate regression using historical official Ireland EPU data
    from sklearn.linear_model import LinearRegression

    X_hist = pca_clean[['weighted_pca_scaled', 'lag1_pca', 'lag2_pca', 'lag3_pca']]
    y_hist = pca_clean['ireland_official']

    model = LinearRegression()
    model.fit(X_hist, y_hist)

    print(f"Regression R² (historical): {model.score(X_hist, y_hist):.4f}")
    print(f"Coefficients: PCA={model.coef_[0]:.4f}, Lag1={model.coef_[1]:.4f}, Lag2={model.coef_[2]:.4f}, Lag3={model.coef_[3]:.4f}")

    # Calculate residuals for historical data to get scaling parameters
    pca_clean = pca_clean.copy()
    pca_clean['predicted'] = model.predict(X_hist)
    pca_clean['residual'] = pca_clean['ireland_official'] - pca_clean['predicted']

    resid_mean = pca_clean['residual'].mean()
    resid_std = pca_clean['residual'].std()
    print(f"Historical residual stats: mean={resid_mean:.2f}, std={resid_std:.2f}")

    # Now apply to scraped months
    # Merge scraped data with PCA data
    pca_df['month'] = pca_df['date'].dt.to_period('M').dt.to_timestamp()
    main_epu_df['month_dt'] = pd.to_datetime(main_epu_df['month'])

    merged = main_epu_df.merge(
        pca_df[['month', 'weighted_pca_scaled', 'lag1_pca', 'lag2_pca', 'lag3_pca']],
        left_on='month_dt',
        right_on='month',
        how='left'
    )

    # For months where we have PCA data, calculate domestic EPU
    has_pca = merged.dropna(subset=['weighted_pca_scaled', 'lag1_pca', 'lag2_pca', 'lag3_pca'])

    if len(has_pca) == 0:
        print("\nNo scraped months have PCA data available.")
        return None

    print(f"\nScraped months with PCA data: {len(has_pca)}")

    # Calculate predicted "foreign-driven" EPU using our scaled scraped EPU
    has_pca = has_pca.copy()
    X_new = has_pca[['weighted_pca_scaled', 'lag1_pca', 'lag2_pca', 'lag3_pca']]
    has_pca['foreign_component'] = model.predict(X_new)

    # Domestic EPU = Actual EPU - Foreign-driven component
    # Since our scraped EPU is scaled, we calculate residual similarly
    has_pca['domestic_raw'] = has_pca['epu_scaled'] - has_pca['foreign_component']

    # Scale domestic EPU to historical residual distribution, then to standard EPU scale
    has_pca['domestic_epu'] = (has_pca['domestic_raw'] - resid_mean) / resid_std * NEW_STDEV + NEW_MEAN

    print(f"\nDomestic EPU calculated for {len(has_pca)} months:")
    for _, row in has_pca.iterrows():
        print(f"  {row['month_x']}: Main={row['epu_scaled']:.2f}, Domestic={row['domestic_epu']:.2f}")

    return has_pca[['month_x', 'epu_scaled', 'domestic_epu']].rename(
        columns={'month_x': 'month', 'epu_scaled': 'main_epu'}
    )


# =============================================================================
# OUTPUT GENERATION
# =============================================================================

def generate_output(main_epu_df, domestic_epu_df=None):
    """Generate final output files with both main and domestic EPU."""
    print("\n" + "=" * 70)
    print("GENERATING OUTPUT FILES")
    print("=" * 70)

    # Filter to June 2025 - January 2026
    main_epu_df['date'] = pd.to_datetime(main_epu_df['month'])
    filtered = main_epu_df[
        (main_epu_df['date'] >= '2025-06-01') &
        (main_epu_df['date'] <= '2026-01-31')
    ].copy()

    # Sort by date
    filtered = filtered.sort_values('date')

    # Prepare output dataframe
    output = filtered[['month', 'date', 'total_articles', 'matched_articles',
                       'share_raw', 'share_scaled', 'epu_scaled']].copy()
    output = output.rename(columns={'epu_scaled': 'ireland_main_epu'})

    # Add domestic EPU if available
    if domestic_epu_df is not None and len(domestic_epu_df) > 0:
        output = output.merge(
            domestic_epu_df[['month', 'domestic_epu']],
            on='month',
            how='left'
        )
        output = output.rename(columns={'domestic_epu': 'ireland_domestic_epu'})
    else:
        # Domestic EPU not available - add placeholder
        output['ireland_domestic_epu'] = np.nan
        print("\nNote: Domestic EPU not calculated (foreign EPU data not available/updated)")

    # Create quarterly aggregation
    output['quarter'] = output['date'].dt.to_period('Q')
    quarterly = output.groupby('quarter').agg({
        'ireland_main_epu': 'mean',
        'ireland_domestic_epu': 'mean'
    }).reset_index()
    quarterly['quarter'] = quarterly['quarter'].astype(str)

    # Display results
    print("\n" + "-" * 70)
    print("FINAL EPU SERIES (June 2025 - January 2026)")
    print("-" * 70)
    print(f"{'Month':<12} {'Main EPU':>12} {'Domestic EPU':>14}")
    print("-" * 40)
    for _, row in output.iterrows():
        dom_str = f"{row['ireland_domestic_epu']:.2f}" if pd.notna(row['ireland_domestic_epu']) else "N/A"
        print(f"{row['month']:<12} {row['ireland_main_epu']:>12.2f} {dom_str:>14}")

    print("\n" + "-" * 70)
    print("QUARTERLY AVERAGES")
    print("-" * 70)
    for _, row in quarterly.iterrows():
        dom_str = f"{row['ireland_domestic_epu']:.2f}" if pd.notna(row['ireland_domestic_epu']) else "N/A"
        print(f"{row['quarter']}: Main={row['ireland_main_epu']:.2f}, Domestic={dom_str}")

    # Save to Excel
    output_file = OUTPUT_DIR / 'ireland_epu_jun2025_jan2026.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Summary
        summary = pd.DataFrame({
            'Item': ['Generated', 'Period', 'Scaling Factor', 'Sources', 'Total Articles', 'Total Matches', 'Methodology'],
            'Value': [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'June 2025 - January 2026',
                str(SCALING_FACTOR),
                'Irish Times, Irish Examiner (Irish Independent blocked)',
                int(output['total_articles'].sum()),
                int(output['matched_articles'].sum()),
                'Web scraping with Boolean search, scaled to match ProQuest methodology'
            ]
        })
        summary.to_excel(writer, sheet_name='Summary', index=False)

        # Monthly data
        output_cols = ['month', 'total_articles', 'matched_articles', 'share_scaled',
                       'ireland_main_epu', 'ireland_domestic_epu']
        output[output_cols].to_excel(writer, sheet_name='Monthly_EPU', index=False)

        # Quarterly data
        quarterly.to_excel(writer, sheet_name='Quarterly_EPU', index=False)

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

    print(f"\n\nOutput saved to: {output_file.resolve()}")

    # Also save CSV
    csv_file = OUTPUT_DIR / 'ireland_epu_jun2025_jan2026.csv'
    output[output_cols].to_csv(csv_file, index=False)
    print(f"CSV saved to: {csv_file.resolve()}")

    return output


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

    # Calculate main EPU
    main_epu = calculate_main_epu()

    # Calculate domestic EPU
    domestic_epu = calculate_domestic_epu(main_epu)

    # Generate output
    output = generate_output(main_epu, domestic_epu)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    return output


if __name__ == "__main__":
    main()
