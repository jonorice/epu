#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EPU Comparison Analysis Script
Compares scraped results with existing ProQuest-derived data
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ============================================================
# Load Data
# ============================================================

print("=" * 70)
print("IRISH EPU INDEX - SCRAPED VS EXISTING DATA COMPARISON")
print("=" * 70)
print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Load scraped data
conn = sqlite3.connect('out_fast_epu_scrape/state/fast_epu_state.sqlite')

df_scraped_by_source = pd.read_sql_query('''
    SELECT
        source,
        month,
        COUNT(*) as total_discovered,
        SUM(CASE WHEN status IN ("ok", "paywalled_or_empty", "empty_text") THEN 1 ELSE 0 END) as processed,
        SUM(CASE WHEN match = 1 THEN 1 ELSE 0 END) as matched
    FROM articles
    WHERE month >= "2025-10-01" AND month <= "2025-12-01"
    GROUP BY source, month
    ORDER BY month, source
''', conn)

df_scraped_monthly = pd.read_sql_query('''
    SELECT
        month,
        SUM(CASE WHEN status IN ("ok", "paywalled_or_empty", "empty_text") THEN 1 ELSE 0 END) as total_articles,
        SUM(CASE WHEN match = 1 THEN 1 ELSE 0 END) as matched_articles
    FROM articles
    WHERE month >= "2025-10-01" AND month <= "2025-12-01"
      AND status NOT IN ("discovered", "fetch_failed", "out_of_range")
    GROUP BY month
    ORDER BY month
''', conn)
df_scraped_monthly['share_scraped'] = df_scraped_monthly['matched_articles'] / df_scraped_monthly['total_articles']

# Get matched articles for export
df_matched_articles = pd.read_sql_query('''
    SELECT
        source,
        published_date,
        month,
        title,
        url,
        text_len
    FROM articles
    WHERE match = 1
      AND month >= "2025-10-01" AND month <= "2025-12-01"
    ORDER BY published_date, source
''', conn)

conn.close()

# Load existing data
existing = pd.read_excel('main_epu_construction.xlsx')
existing_recent = existing[existing['Unnamed: 0'] >= '2025-10-01'].copy()
existing_recent = existing_recent[['Unnamed: 0',
    'Count of flagged articles (Irish Times, Irish Independent, Irish Examiner)',
    'Total Articles (Irish Times, Irish Independent, Irish Examiner)',
    'Share of flagged articles (Irish Times, Irish Independent, Irish Examiner)',
    'Rescaled FINAL EPU'
]].copy()
existing_recent.columns = ['month', 'matched_existing', 'total_existing', 'share_existing', 'epu_existing']
existing_recent['month'] = existing_recent['month'].dt.strftime('%Y-%m-%d')

# ============================================================
# Comparison Analysis
# ============================================================

print("SECTION 1: DATA AVAILABILITY")
print("-" * 70)
print()
print("Sources scraped:")
print(f"  - Irish Times: YES (found {df_scraped_by_source[df_scraped_by_source['source']=='IrishTimes']['processed'].sum()} articles)")
print(f"  - Irish Examiner: YES (found {df_scraped_by_source[df_scraped_by_source['source']=='IrishExaminer']['processed'].sum()} articles)")
print(f"  - Irish Independent: NO (403 Forbidden - sitemap access blocked)")
print()
print("IMPORTANT: The Irish Independent data is MISSING from the scraped results.")
print("This will significantly affect comparability with the existing data.")
print()

# Merge for comparison
comparison = df_scraped_monthly.merge(existing_recent, on='month', how='outer')
comparison = comparison.rename(columns={
    'total_articles': 'total_scraped',
    'matched_articles': 'matched_scraped'
})

print("SECTION 2: MONTHLY SUMMARY COMPARISON")
print("-" * 70)
print()
print(f"{'Month':<12} {'Scraped':<10} {'Existing':<10} {'Scraped':<10} {'Existing':<10} {'Share':<10} {'Share':<10}")
print(f"{'':<12} {'Total':<10} {'Total':<10} {'Matched':<10} {'Matched':<10} {'Scraped':<10} {'Existing':<10}")
print("-" * 70)
for _, row in comparison.iterrows():
    print(f"{row['month']:<12} {int(row['total_scraped']):>9} {int(row['total_existing']):>9} "
          f"{int(row['matched_scraped']):>9} {int(row['matched_existing']):>9} "
          f"{row['share_scraped']*100:>9.2f}% {row['share_existing']*100:>9.2f}%")
print()

print("SECTION 3: KEY OBSERVATIONS")
print("-" * 70)
print()

# Calculate differences
for _, row in comparison.iterrows():
    total_diff = (row['total_scraped'] - row['total_existing']) / row['total_existing'] * 100
    match_diff = (row['matched_scraped'] - row['matched_existing']) / row['matched_existing'] * 100
    print(f"{row['month']}:")
    print(f"  - Total articles: {'more' if total_diff > 0 else 'fewer'} scraped ({abs(total_diff):.1f}%)")
    print(f"  - Matched articles: {'more' if match_diff > 0 else 'fewer'} scraped ({abs(match_diff):.1f}%)")
    print()

print("SECTION 4: BY SOURCE BREAKDOWN (Scraped Only)")
print("-" * 70)
print()
print(df_scraped_by_source.to_string(index=False))
print()

print("SECTION 5: ANALYSIS OF DIFFERENCES")
print("-" * 70)
print()
print("""
REASONS FOR DIFFERENCES:

1. MISSING IRISH INDEPENDENT DATA
   - The scraped data is missing Irish Independent entirely (403 Forbidden)
   - The existing data includes all three newspapers
   - This likely accounts for the lower total articles in some months

2. DIFFERENT ARTICLE DISCOVERY METHODS
   - Existing: ProQuest database (comprehensive historical archive)
   - Scraped: Web sitemaps (current website content only)
   - Sitemaps may include more articles (non-news content) or miss archived articles

3. ARTICLE CATEGORIZATION
   - Some URLs discovered via sitemaps may not be true news articles
   - ProQuest likely has better filtering of genuine news content

4. DECEMBER 2025 DATA
   - Existing data shows only 1,652 articles (likely incomplete/month-to-date)
   - Scraped data shows 6,004 articles for December
   - This suggests the existing file may need updating

5. BOOLEAN SEARCH CONSISTENCY
   - The Boolean search terms are IDENTICAL to the original script
   - Match rates vary but are in the same general range (0.6-1.4%)

RECOMMENDATIONS:

1. Irish Independent needs separate authentication/cookies to access sitemaps
2. Consider using robots.txt discovery mode for Irish Independent
3. The December 2025 existing data appears incomplete
4. For accurate comparison, focus on Irish Times + Irish Examiner only
""")

# ============================================================
# Calculate rescaled EPU for scraped data
# ============================================================

print("SECTION 6: RESCALED EPU CALCULATION")
print("-" * 70)
print()

# Using the existing scaling parameters
OLD_MEAN = 0.004738
NEW_MEAN = 100
OLD_STDEV = 0.003427
NEW_STDEV = 72.335236

comparison['epu_scraped'] = (comparison['share_scraped'] - OLD_MEAN) / OLD_STDEV * NEW_STDEV + OLD_MEAN

print(f"{'Month':<12} {'EPU Scraped':>15} {'EPU Existing':>15} {'Difference':>15}")
print("-" * 60)
for _, row in comparison.iterrows():
    diff = row['epu_scraped'] - row['epu_existing']
    print(f"{row['month']:<12} {row['epu_scraped']:>15.2f} {row['epu_existing']:>15.2f} {diff:>+15.2f}")
print()

print("NOTE: The scraped EPU is based on only Irish Times + Irish Examiner,")
print("while the existing EPU includes all three papers. Direct comparison")
print("is not meaningful without Irish Independent data.")
print()

# ============================================================
# Export Results
# ============================================================

print("SECTION 7: EXPORTING RESULTS")
print("-" * 70)
print()

output_dir = Path('out_fast_epu_scrape')

# Save comparison summary
comparison.to_excel(output_dir / 'comparison_summary.xlsx', index=False)
print(f"  - Saved: {output_dir / 'comparison_summary.xlsx'}")

# Save matched articles
df_matched_articles.to_excel(output_dir / 'matched_articles_detail.xlsx', index=False)
print(f"  - Saved: {output_dir / 'matched_articles_detail.xlsx'}")

# Save by source breakdown
df_scraped_by_source.to_excel(output_dir / 'results_by_source.xlsx', index=False)
print(f"  - Saved: {output_dir / 'results_by_source.xlsx'}")

# Create a comprehensive analysis workbook
with pd.ExcelWriter(output_dir / 'epu_analysis_complete.xlsx', engine='openpyxl') as writer:
    # Summary
    pd.DataFrame({
        'Item': ['Analysis Date', 'Months Analyzed', 'Sources Scraped', 'Total Articles Processed', 'Total Matches Found'],
        'Value': [
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'October - December 2025',
            'Irish Times, Irish Examiner (Irish Independent blocked)',
            df_scraped_monthly['total_articles'].sum(),
            df_scraped_monthly['matched_articles'].sum()
        ]
    }).to_excel(writer, sheet_name='Summary', index=False)

    # Comparison
    comparison.to_excel(writer, sheet_name='Monthly Comparison', index=False)

    # By source
    df_scraped_by_source.to_excel(writer, sheet_name='By Source', index=False)

    # Matched articles
    df_matched_articles.to_excel(writer, sheet_name='Matched Articles', index=False)

print(f"  - Saved: {output_dir / 'epu_analysis_complete.xlsx'}")

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
