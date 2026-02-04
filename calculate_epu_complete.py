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
    try:
        foreign_epu = pd.read_excel('dataset_est_domestic.xlsx')
    except FileNotFoundError:
        print("WARNING: dataset_est_domestic.xlsx not found.")
        print("Cannot calculate domestic EPU without foreign country data.")
        return None

    print(f"\nLoaded foreign EPU data: {foreign_epu.shape[0]} rows, {foreign_epu.shape[1]} columns")

    # Get column names
    date_col = foreign_epu.columns[0]  # First column is date

    # Columns to exclude (as per README)
    exclude_cols = [date_col, 'Ireland_Rice', 'GEPU_current', 'GEPU_ppp',
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

    # Get Ireland_Rice from original data if available
    if 'Ireland_Rice' in foreign_epu.columns:
        ireland_epu = foreign_epu['Ireland_Rice'].values
        gepu_mean = np.nanmean(ireland_epu)
        gepu_std = np.nanstd(ireland_epu)
        pca_df['weighted_pca_scaled'] = (pca_df['weighted_pca'] - pca_mean) / pca_std * gepu_std + gepu_mean
    else:
        pca_df['weighted_pca_scaled'] = (pca_df['weighted_pca'] - pca_mean) / pca_std * NEW_STDEV + NEW_MEAN

    # Now merge with main EPU data
    main_epu_df['date'] = pd.to_datetime(main_epu_df['month'])

    # For the months we have scraped data, we'll calculate domestic EPU
    # We need to extend the weighted PCA series or use only overlapping periods

    # Merge on date (monthly)
    pca_df['month'] = pca_df['date'].dt.to_period('M').dt.to_timestamp()
    main_epu_df['month_dt'] = pd.to_datetime(main_epu_df['month'])

    # Check overlap
    scraped_months = set(main_epu_df['month_dt'])
    pca_months = set(pca_df['month'])
    overlap = scraped_months.intersection(pca_months)

    print(f"\nScraped months: {len(scraped_months)}")
    print(f"PCA data months: {len(pca_months)}")
    print(f"Overlapping months: {len(overlap)}")

    if len(overlap) == 0:
        print("\nWARNING: No overlapping months between scraped data and foreign EPU data.")
        print("The foreign EPU data ends at: ", pca_df['date'].max())
        print("Scraped data starts at: ", main_epu_df['date'].min())
        print("\nTo calculate domestic EPU, you need to:")
        print("1. Download updated country-level data from policyuncertainty.com")
        print("2. Update dataset_est_domestic.xlsx with recent months")

        # Return None for domestic EPU, but we can still estimate using a simplified approach
        print("\nUsing simplified domestic calculation (without foreign EPU adjustment)...")

        # Simple approach: domestic = main EPU (no foreign adjustment available)
        # In practice, you'd want the full methodology with updated foreign data
        return None

    # If we have overlap, proceed with regression
    merged = main_epu_df.merge(
        pca_df[['month', 'weighted_pca_scaled']],
        left_on='month_dt',
        right_on='month',
        how='inner'
    )

    if len(merged) < 5:
        print(f"\nInsufficient overlapping data ({len(merged)} months) for regression.")
        return None

    # Create lagged variables (3 lags as per R script)
    merged = merged.sort_values('month_dt')
    merged['lag1_pca'] = merged['weighted_pca_scaled'].shift(1)
    merged['lag2_pca'] = merged['weighted_pca_scaled'].shift(2)
    merged['lag3_pca'] = merged['weighted_pca_scaled'].shift(3)

    # Remove NaN rows from lagging
    merged_clean = merged.dropna(subset=['lag1_pca', 'lag2_pca', 'lag3_pca'])

    if len(merged_clean) < 5:
        print(f"\nInsufficient data after lagging ({len(merged_clean)} months).")
        return None

    # Run regression: Irish_EPU ~ weighted_pca + lag1 + lag2 + lag3
    from sklearn.linear_model import LinearRegression

    X = merged_clean[['weighted_pca_scaled', 'lag1_pca', 'lag2_pca', 'lag3_pca']]
    y = merged_clean['epu_scaled']

    model = LinearRegression()
    model.fit(X, y)

    # Get residuals = domestic uncertainty
    merged_clean['domestic_epu'] = y - model.predict(X)

    # Scale domestic EPU to have similar properties to main EPU
    domestic_mean = merged_clean['domestic_epu'].mean()
    domestic_std = merged_clean['domestic_epu'].std()
    merged_clean['domestic_epu_scaled'] = (merged_clean['domestic_epu'] - domestic_mean) / domestic_std * NEW_STDEV + NEW_MEAN

    print(f"\nRegression R²: {model.score(X, y):.4f}")
    print(f"Domestic EPU calculated for {len(merged_clean)} months")

    return merged_clean[['month_x', 'epu_scaled', 'domestic_epu_scaled']].rename(
        columns={'month_x': 'month', 'epu_scaled': 'main_epu', 'domestic_epu_scaled': 'domestic_epu'}
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
