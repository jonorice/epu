# Irish EPU Update Workflow

This repo maintains three deliverables:

- `main_epu_construction.xlsx`: working sheet for the Irish main EPU series.
- `dataset_est_domestic.xlsx`: processed country panel used to estimate the domestic residual series.
- `Ireland_Policy_Uncertainty_Data_Rice.xlsx`: public-facing output file with the main and domestic Irish EPU series plus the citation link.

## Recommended path

Use `update_latest_epu.py`.

That script is the maintained update path now. It automates the workflow that proved most reliable in practice:

1. scrape Irish Times counts for an overlap anchor month, the latest completed month, and the current partial month;
2. recover the combined-series anchor share from the existing published output;
3. extend the main series using Irish Times flagged-share growth;
4. refresh the domestic input panel from the latest all-country file on `policyuncertainty.com`;
5. rebuild the domestic residual series and rewrite the three workbook outputs.

Example:

```powershell
python update_latest_epu.py --anchor-month 2026-01 --full-month 2026-02 --partial-end 2026-03-11
```

What those arguments mean:

- `--anchor-month`: last full month in the current public output that we trust as the combined-series overlap anchor.
- `--full-month`: latest completed month to update.
- `--partial-end`: last day to include for the current partial month.

The script updates these files in place:

- `main_epu_construction.xlsx`
- `dataset_est_domestic.xlsx`
- `Ireland_Policy_Uncertainty_Data_Rice.xlsx`

By default it also creates a temporary scratch directory for resumable Irish Times scraping and removes it again after a successful run. Use `--keep-scratch` if you want to inspect or reuse that state.

## Why the updater uses the fallback

The original multi-paper archive workflow is no longer consistently reliable. In recent updates:

- Irish Times could still be scraped with acceptable stability.
- Irish Examiner and Irish Independent were more brittle because of paywall and archive issues.

So the maintained process is:

- use Irish Times observed shares for the overlap/full/partial months;
- project the combined main-series share from the overlap month using Irish Times growth;
- then rerun the domestic estimation from the latest all-country file.

This is the path the new script implements.

## Domestic series notes

The latest all-country file on `policyuncertainty.com` does not perfectly match the older local domestic input sheet:

- the website file no longer carries `Netherlands`;
- it does carry `Pakistan`;
- China coverage also changed relative to the older local processed file.

To keep the domestic series updateable, `update_latest_epu.py`:

- uses the existing local processed history as the base;
- appends the newest rows from the website file;
- carries `Pakistan`;
- drops `Netherlands`;
- bridges the China input using the website columns that are still available;
- rescales the rebuilt domestic raw residual back onto the published domestic-series scale using the historical overlap in `Ireland_Policy_Uncertainty_Data_Rice.xlsx`.

## Supporting scripts

- `update_latest_epu.py`: maintained end-to-end updater.
- `1m_individual_papers.py`: low-level article scraper helper. It is still useful, but its file defaults are sample values, not the recommended update path.
- `create_dom_epu.r`: legacy exploratory PCA/regression script kept for reference only. It is not the maintained update path.

## Practical rule for future updates

When asked to "update the latest main and domestic EPU":

1. choose the last full published month as `--anchor-month`;
2. choose the latest completed calendar month as `--full-month`;
3. choose today's date for `--partial-end`;
4. run `update_latest_epu.py`;
5. report the Irish Times flagged/total counts for the anchor, full, and partial months;
6. report the latest main EPU month and the latest domestic EPU month.
