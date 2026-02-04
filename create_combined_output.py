#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create combined EPU output file with all scraped months (Jan, Oct-Dec 2025)
"""

import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = Path("out_fast_epu_scrape")
DB_PATH = OUTPUT_DIR / "state" / "fast_epu_state.sqlite"

# Rescaling parameters from existing data
OLD_MEAN = 0.004738
OLD_STDEV = 0.003427
NEW_MEAN = 100
NEW_STDEV = 72.335236

# =============================================================================
# Load and process data
# =============================================================================

print("=" * 70)
print("COMBINED EPU OUTPUT - ALL SCRAPED MONTHS")
print("=" * 70)
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

conn = sqlite3.connect(str(DB_PATH))

# Monthly summary
df_monthly = pd.read_sql_query("""
    SELECT
        month,
        SUM(CASE WHEN status IN ('ok', 'paywalled_or_empty', 'empty_text') THEN 1 ELSE 0 END) as total_articles,
        SUM(CASE WHEN match = 1 THEN 1 ELSE 0 END) as matched_articles,
        SUM(CASE WHEN paywalled = 1 THEN 1 ELSE 0 END) as paywalled_articles
    FROM articles
    WHERE month IS NOT NULL
      AND status NOT IN ('discovered', 'fetch_failed', 'out_of_range')
    GROUP BY month
    ORDER BY month
""", conn)

df_monthly['share'] = df_monthly['matched_articles'] / df_monthly['total_articles']
df_monthly['rescaled_epu'] = (df_monthly['share'] - OLD_MEAN) / OLD_STDEV * NEW_STDEV + NEW_MEAN

# By source
df_by_source = pd.read_sql_query("""
    SELECT
        source,
        month,
        COUNT(*) as total,
        SUM(CASE WHEN match = 1 THEN 1 ELSE 0 END) as matched,
        SUM(CASE WHEN paywalled = 1 THEN 1 ELSE 0 END) as paywalled
    FROM articles
    WHERE month IS NOT NULL
      AND status NOT IN ('discovered', 'fetch_failed', 'out_of_range')
    GROUP BY source, month
    ORDER BY month, source
""", conn)

# All matched articles
df_matches = pd.read_sql_query("""
    SELECT
        source,
        published_date,
        month,
        title,
        url,
        text_len,
        econ_hit,
        uncertainty_hit,
        policy_hit
    FROM articles
    WHERE match = 1
    ORDER BY published_date, source
""", conn)

conn.close()

# =============================================================================
# Display results
# =============================================================================

print("MONTHLY SUMMARY")
print("-" * 70)
print(f"{'Month':<12} {'Total':>8} {'Matched':>10} {'Share':>10} {'EPU':>10}")
print("-" * 70)
for _, row in df_monthly.iterrows():
    print(f"{row['month']:<12} {int(row['total_articles']):>8} {int(row['matched_articles']):>10} "
          f"{row['share']*100:>9.2f}% {row['rescaled_epu']:>10.2f}")

print()
print("BY SOURCE BREAKDOWN")
print("-" * 70)
pivot = df_by_source.pivot_table(index='month', columns='source', values='matched', aggfunc='sum', fill_value=0)
print(pivot.to_string())

print()
print(f"TOTAL MATCHED ARTICLES: {len(df_matches)}")

# =============================================================================
# Export to Excel
# =============================================================================

output_file = OUTPUT_DIR / "epu_combined_all_months.xlsx"

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # Summary info
    summary = pd.DataFrame({
        'Item': [
            'Generated',
            'Months Covered',
            'Sources',
            'Total Articles Processed',
            'Total Matches',
            'Note'
        ],
        'Value': [
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'January 2025, October-December 2025',
            'Irish Times, Irish Examiner (Irish Independent blocked)',
            int(df_monthly['total_articles'].sum()),
            int(df_monthly['matched_articles'].sum()),
            'Irish Independent sitemaps blocked (403). Results only from IT + IE.'
        ]
    })
    summary.to_excel(writer, sheet_name='Summary', index=False)

    # Monthly summary with EPU
    df_monthly.to_excel(writer, sheet_name='Monthly EPU', index=False)

    # By source breakdown
    df_by_source.to_excel(writer, sheet_name='By Source', index=False)

    # All matched articles
    df_matches.to_excel(writer, sheet_name='Matched Articles', index=False)

    # Rescaling parameters
    params = pd.DataFrame({
        'Parameter': ['OLD_MEAN', 'OLD_STDEV', 'NEW_MEAN', 'NEW_STDEV', 'Formula'],
        'Value': [OLD_MEAN, OLD_STDEV, NEW_MEAN, NEW_STDEV,
                  '(share - OLD_MEAN) / OLD_STDEV * NEW_STDEV + NEW_MEAN']
    })
    params.to_excel(writer, sheet_name='Rescaling Parameters', index=False)

print()
print(f"Output saved to: {output_file.resolve()}")

# Also save CSV versions
df_monthly.to_csv(OUTPUT_DIR / "monthly_epu_all.csv", index=False)
df_matches.to_csv(OUTPUT_DIR / "matched_articles_all.csv", index=False)
print(f"CSV files saved to: {OUTPUT_DIR.resolve()}")

print()
print("=" * 70)
print("DONE")
print("=" * 70)
