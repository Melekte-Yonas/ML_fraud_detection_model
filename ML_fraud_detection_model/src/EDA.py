#EDA Goal: Find best columns for fraud predictors and clean it
#Dataset: Bank_fraud_dataset(100k rows)

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

#loaded the data
def load_data():
    data_path = BASE_DIR / "Data" / "raw" / "Bank_fraud_dataset"
    df = pd.read_csv(data_path)
    return df

df = load_data()
pd.set_option("display.max_columns", None)

#I created these two function to find good indicators for fraud.
#I found that I had to use these often to get a clear understanding of the data.

#This function groups by the column's categories and finds the average fraud within each category
#I used this for the categorial columns.
def mean1(df, column_name):
    return print(f"\n{df.groupby(column_name)["is_fraud"].mean()}")

#This function splits the column's numbers into 4 categories based on its standard deviation and creates a new categorical column.
#"min-25%", "25%-50%", "50%-75%", "75%-max" ---> then I used the first function to find the average fraud rate within each category.
def bin_column(df, column_name):
    labels = ["min-25%", "25%-50%", "50%-75%", "75%-max"]
    bins = [df[column_name].quantile(0),
            df[column_name].quantile(0.25),
            df[column_name].quantile(0.50),
            df[column_name].quantile(0.75),
            df[column_name].quantile(1.0)]
    new_column = f"{column_name}_bins"
    df[new_column] = pd.cut(df[column_name],
                        bins=bins,
                        labels=labels
    )
    return new_column



#I. CLEANING THE DATA

#Kept columns that I considered useful. Kept behavioural data and dropped identifiers, and columns that cause data leakage.
df = df[["hour_of_day",
         "is_weekend",
         "country",
         "city",
         "merchant_category",
         "payment_method",
         "device_type",
         'account_age_years',
         "account_balance",
         "transaction_amount",
         "num_prev_transactions",
         "time_since_last_txn_hrs",
         "failed_attempts",
         "distance_from_home_km",
         "is_international",
         "pin_changed_recently",
         "is_fraud",
         "is_night_transaction",
         "transaction_freq_monthly",
         ]]

#No null values and fraud rate is 5.5%
print(df.info())
print(f"\n{(df["is_fraud"].mean())*100:.2f}% is fraud\n")


#II. FINDING USEFUL COLUMNS FOR FRAUD LABELLING

#1. column: "hour_of_day"
print(f"\n{df["hour_of_day"].describe()}\n")
bin_column(df, "hour_of_day")
mean1(df, "hour_of_day_bins")
print()
#results show that times from 1am till 6am contain the most fraud at 8% compared to the average of 5%
#not the best indicator for fraud but definitely useful.


#2. column: "is_weekend"
mean1(df, "is_weekend")
print()
#Almost no difference makes it pretty bad indicator of fraud.


#3. column: country
mean1(df, "country")
print()
#no significant difference makes it pretty bad indicator of fraud


#4. column: city
mean1(df, "city")
print()
#cities with higher population have higher fraud rates(LA, New York, Paris, e.t.c)
#still not a good indicator

#5. merchant category
mean1(df, "merchant_category")
print()
#This is a good indicator certain categories contain higher than average fraud rates.
#ATM Withdrawal, Crypto Exchange, and Jewelery clock in at 8% fraud rate about 3% more than the average for the data.

#6. payment_method
mean1(df, "payment_method")
print()
#not really a good indicator but cheque is at about 6% fraud rate, the highest in this column.

#7. device_type
mean1(df, "device_type")
print()
#pretty bad indicator, all the categories hover around 5%.

#8. account_age_years
bin_column(df, "account_age_years")
mean1(df, "account_age_years_bins")
print()
#Account age is not a good indicator either.


#9. column: account_balance
bin_column(df, "account_balance")
mean1(df, "account_balance_bins")
print()
#The results are similar, no specific range of balance indicates fraud


#10. column: transaction_amount
bin_column(df, "transaction_amount")
mean1(df, "transaction_amount_bins")
print()
#same thing here not a good indicator


#11. column: "num_prev_transactions"
bin_column(df, "num_prev_transactions")
mean1(df, "num_prev_transactions_bins")
print()
#not a good indicator as well


#12. column: "time_since_last_txn_hrs"
print(F"\n{df["time_since_last_txn_hrs"].describe()}")
bin_column(df, "time_since_last_txn_hrs")
mean1(df, "time_since_last_txn_hrs_bins")
print()
#hours 0-3 have the highest at about 6%
#not the best, but it's still something


#13. column: "failed_attempts":
mean1(df, "failed_attempts")
print()
# 2, 3, 4, and 5 failed attempts yield about 14-15% fraud.
#The best indicator so far, more than double the average for the whole data set.


#14. column: "distance_from_home_km":
bin_column(df, "distance_from_home_km")
mean1(df, "distance_from_home_km_bins")
print()
# not a good indicator of fraud


#15. column: "is_international":
mean1(df, "is_international")
print()
#international transactions have 10% fraud rate, double the average while non-international are about 4.6%,
#pretty good indicator of fraud


#16. column: "pin_changed_recently":
mean1(df, "pin_changed_recently")
print()
#recently changed pin accounts have about 8% fraud rate.
#an okay indicator of fraud


#17. column: "is_night_transaction":
mean1(df, "is_night_transaction")
print()
#night transactions have about 8% fraud rate, better than nothing


#18. Column: "transaction_freq_monthly":
bin_column(df, "transaction_freq_monthly")
mean1(df, "transaction_freq_monthly_bins")
print()
# not a useful fraud indicator


df = df[["hour_of_day",
         "hour_of_day_bins",
         "is_weekend",
         "city",
         "merchant_category",
         "payment_method",
         "account_age_years",
         "time_since_last_txn_hrs",
         "time_since_last_txn_hrs_bins",
         "failed_attempts",
         "is_international",
         "pin_changed_recently",
         "is_night_transaction",
         "is_fraud"]]
#Included the columns that were the best indicators and dropped everything else. started of with 16 ended it with 13.
#This data set did not really have good indicators. It was a synthetic data. This was the best I found.
#There aren't really any real transaction datasets due to it being customer sensitive information.



#Created a new dataset based on these columns.
df.to_csv(BASE_DIR / "Data/processed/eda_cleaned_sample.csv")
