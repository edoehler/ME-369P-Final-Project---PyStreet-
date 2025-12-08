# ME-369P-Final-Project---PyStreet-

This repository contains our ME 369P final project, PyStreet, where we use pandas and other packages to build a full event-study pipeline for earnings announcements. The main goal is to show how far you can go with data wrangling, grouping, merging, and analysis in pandas, from raw API calls all the way to cleaned event-level data and plots. The second project requirement we satisfy is using an external API to generate sentiment scores for earnings call transcripts.

Here, we conduct an event-study project analyzing earnings announcements for large-cap companies in the Technology, Financials, and Health Care sectors. We focus on abnormal returns around each earnings event, specifically looking at:
- pre-earnings drift (days −5 to −1)
- the event-day reaction (day 0)
- post-earnings drift (days +1 to +5). 

The repository includes a Python script (Data_Extraction.py) for data extraction. This code  uses the Financial Modeling Prep (FMP) API to pull:
- 5 years of daily price data for each stock
- earnings calendar data for each stock (earnings dates, actual EPS, estimated EPS)
- 5 years of S&P 500 daily returns
  
**This script (Data_Extraction.py) requires an FMP API key, which is not included to protect a group member's subscription**, so it will not run without you adding your own key. We also provide a finalized dataset (FINAL_DATA.csv) containing the cleaned event-level dataframe with all variables needed for analysis so you can play around with the numbers without an API key. The FINAL_DATA.csv is the output of Data_Extraction.py when a API key is present. Additionally, there is a folder of earnings call transcripts (transcripts.zip) for each event (stored as .txt or .docx files and bundled in a zip), which are used to compute a sentiment score with FinBERT. 

Finally, a separate Python script (Data_Analysis.py) performs the data analysis using pandas and plotting packages, and generates visualizations to summarize how prices react to surprises and management tone; this analysis script can be run directly as long as the final dataset file (FINAL_DATA.csv) is present.  

The following is our key takeaways from the graphs and analysis done in the code. All three sectors exhibit some degree of information leakage, with Financials standing out the most: prices tend to drift in the “correct” direction before earnings, especially ahead of negative surprises. For Technology, we find a clear positive relationship between EPS surprise and day-of abnormal returns (bigger beats → bigger pops), but the average drift from days +1 to +5 is negative across the board. In other words, even when the initial reaction is positive, the next few days tend to give some of it back. We also see that beats and big beats are extremely common, which suggests that analyst estimates are conservative; as a result, the market really only “rewards” companies when the beat is unusually large, while small beats barely move the needle. EPS predictions are often very close to real quartly EPS announcements - this is both a product of proper due diligence by analysts, appropriate disclosures from companies, and managers driving the meet EPS expectations.

When we slice the data by year, we find that day-of abnormal returns have actually flipped negative on average over the past two years, highlighting a more cautious market reaction to earnings. Statistically, only one sector shows a significant correlation between day-0 reactions and subsequent drift, which implies that, in our sample, markets are mostly efficient with respect to earnings news. Overall, the five-day post-earnings drift is negative on average, so earnings releases are not a great time to buy stocks in general—if anything, they are periods of elevated volatility rather than easy excess-return opportunities.

Files:

**Data_extraction.py:** This is a super useful file for anybody who has a FMP API key or a similair API that would work with the script. THis file provides a clear framework for accessing API data over a period of time across multiple metrics and stocks. Others could modify this script easily to access data for other stocks, time periods, metrics, etc.

**Data_Analysis.py:** This is a useful file becasue it crunches the numbers fed into it and outputs plots relevent to Post-Earnings Announcement Drift. This file can easily be modified to output different plots or to protray different metrics.

**FINAL_DATA.csv:** This file is used in Data_Analysis.py if a user does not have an API key. This allows people to analyze the data on their own if they do not use the key, since we saved the data we used in our project there. This file also stores, in addition to stock and PEAD-related data, the FinBERT sentiment scores.

**transcripts.zip:** This is a zip file that has all the earnings call transcripts over the quarters and companies we analyzed in this project. This file is needed for the FinBERT neural network because FinBERT ingests transcripts and outputs a number based on the earnings call. Pulling a large quantity of transcripts can be hard or cost a subscription, so having them all in one place here is integral for the project.

Possible extensions:
- Expand on the timeframe, sectors, or number of companies analyzed
- Look at different event windows(example: -10 to +10 days)
- Look at other models besides CAPM, such as Fama-French
- Continue to refine the FinBERT model by looking at Q&A transcripts as well, see if tokenizing by sentence provides more insight, ect.
- Try other nodels that are not FinBERT to compare model efficacy 

Division of Labor:
- Grant: Pulling data off Financial Modeling Prep, FinBERT testing, and  analyzing data to find insights and data points of interest. 
- Ethan: Earnings transcript sourcing, analyzing data to find insights and data points of interest, LLM / neural network research.
- Ryan: Building abnormal return calculations from raw data, analyzing data to find insights and data points of interest, earnings transcript sourcing.
