# Bank Customer Dataset (`bank_data.csv`)

This dataset contains demographic, financial, account, and behavioral records for **1,000 bank customers**, including deliberate **missing values (`NaN`)** designed for data cleaning, preprocessing, imputation, and machine learning workflows.

---

## File Location
* **Path:** `e:\LANGCHAIN\ML\Files\bank_data.csv`
* **Rows:** 1,000 customers
* **Columns:** 19 features

---

## Data Dictionary

| Column Name | Data Type | Description | Missing Values | Example |
| :--- | :--- | :--- | :--- | :--- |
| `CustomerID` | String | Unique identifier for each customer | 0 (0%) | `CUST_1000` |
| `CustomerName` | String | Full customer name | 0 (0%) | `Aaron Phillips` |
| `Age` | Float / Int | Customer's age (18–78) | ~82 (~8.2%) | `57.0` |
| `Gender` | Categorical | Customer gender (`Male`, `Female`) | ~55 (~5.5%) | `Male` |
| `MaritalStatus` | Categorical | `Married`, `Single`, `Divorced` | ~61 (~6.1%) | `Divorced` |
| `EmploymentStatus` | Categorical | `Employed`, `Self-Employed`, `Unemployed`, `Retired`, `Student` | ~57 (~5.7%) | `Student` |
| `AnnualIncome` | Float | Annual income in USD | ~99 (~9.9%) | `84537.05` |
| `CreditScore` | Float / Int | Credit score rating (350–850) | ~81 (~8.1%) | `731.0` |
| `AccountType` | Categorical | Primary account type (`Checking`, `Savings`, `Investment`, `Money Market`) | ~44 (~4.4%) | `Checking` |
| `AccountBalance` | Float | Current balance in bank account | ~105 (~10.5%) | `148175.40` |
| `NumberOfProducts` | Float / Int | Number of bank products held (1–4) | ~39 (~3.9%) | `2.0` |
| `HasCreditCard` | Float / Bool | Whether customer holds a credit card (`1.0` = Yes, `0.0` = No) | ~51 (~5.1%) | `1.0` |
| `CreditCardBalance` | Float | Outstanding balance on credit card | ~111 (~11.1%) | `1807.67` |
| `IsActiveMember` | Float / Bool | Activity status (`1.0` = Active, `0.0` = Inactive) | ~40 (~4.0%) | `0.0` |
| `EstimatedSalary` | Float | Estimated annual salary | ~79 (~7.9%) | `83579.56` |
| `LoanStatus` | Categorical | Status of loan application (`None`, `Approved`, `Pending`, `Rejected`) | ~54 (~5.4%) | `Pending` |
| `LoanAmount` | Float | Total loan amount in USD | ~84 (~8.4%) | `29147.35` |
| `CreditDefaultRisk`| Categorical | Default risk assessment (`Low`, `Medium`, `High`) | ~64 (~6.4%) | `Low` |
| `Churned` | Float / Bool | Target variable (`1.0` = Exited / Churned, `0.0` = Retained) | ~40 (~4.0%) | `1.0` |

---

## Quick Start with Pandas

```python
import pandas as pd

# 1. Load the dataset
df = pd.read_csv("Files/bank_data.csv")

# 2. Inspect shape and columns
print("Shape:", df.shape)
df.head()

# 3. Check missing values count and percentage
missing_count = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({"Missing Count": missing_count, "Percentage (%)": missing_pct})
print(missing_df[missing_df["Missing Count"] > 0])
```
