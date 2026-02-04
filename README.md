This contains codes required to update two data series, Irish main EPU and Irish domestic EPU. For the main EPU the script 1m_individual_papers.py should pull articles referring to 'economic policy uncertainty' using the boolean search criteria summarised in the pdf paper included in the repository. 
Once these codes have been run, they should be used to populate an additional month in the base file for main EPU called 'main_epu_construction.xlsx' where it is rescaled to constantly enforce the series mean to be 100 and standard deviation to be scaled accordingly (as shown in the excel). 
Next you have to work on producing the domestic EPU series, which is a residual (unexplained) portion of the main EPU after predicting it with a Prinipal component analysis of the foreign series available on the policyuncertainty.com website. 
This requires the following steps: 1) download the country-level data file from: https://www.policyuncertainty.com/all_country_data.html
2) remove the following countries from this country level data file: GEPU_current, GEPU_ppp, Singapore, Ireland, SCMP China,	Mainland China, Sweden, Mexico, delete all data prior to Jan 1997.
3) add the Irish main EPU series created in steps 1-2 above to the all country data file. 
The all country data file (called dataset_est_domestic currently in the repository) is then pulled in to an R code that creates the residual series 
The series is also scaled in line with the main series. 
You should then create one output file resembling the one in the repository called 'Ireland_Policy_Uncertainty_Data_Rice..' except with the latest data available. Ensure a citation link to the paper in Economic and Social Review is also added below the data columns.  




