# ME-369P-Final-Project---PyStreet-

This repository contains our ME 369P final project, PyStreet, where we use pandas to build a full event-study pipeline for earnings announcements. The main goal is to show how far you can go with data wrangling, grouping, merging, and analysis in pandas, from raw API calls all the way to cleaned event-level data and plots. The second project requirement we satisfy is using an external API to generate sentiment scores for earnings call transcripts.

Here we conduct an event-study project analyzing earnings announcements for large-cap companies in the Technology, Financials, and Health Care sectors. We focus on abnormal returns around each earnings event, specifically looking at:
- pre-earnings drift (days −5 to −1)
- the event-day reaction (day 0)
- post-earnings drift (days +1 to +5). 

The repository includes a Python script (Data_Extraction.py) for data extraction that uses the Financial Modeling Prep (FMP) API to pull daily price data and earnings calendar data for each stock; this script requires an FMP API key, which is not included to protect a group member's subscription, so it will not run without you adding your own key. We also provide a finalized dataset (FINAL_DATA.csv) containing the cleaned event-level dataframe with all variables needed for analysis so you can play around with the numbers without an API key. Additionally, there is a folder of earnings call transcripts (transcripts.zip) for each event (stored as .txt or .docx files and bundled in a zip), which are used to compute a sentiment score with FinBERT. 

Finally, a separate Python script (Data_Analysis.py) performs the data analysis using pandas and plotting packages, and generates visualizations to summarize how prices react to surprises and management tone; this analysis script can be run directly as long as the final dataset file (FINAL_DATA.csv) is present.  
