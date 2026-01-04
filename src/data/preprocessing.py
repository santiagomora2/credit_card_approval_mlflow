import pandas as pd

def preprocess_data():
    app_rd = pd.read_csv("./data/raw/application_record.csv")
    cred_rd = pd.read_csv("./data/raw/credit_record.csv")

    # Extract from application_record.csv those IDs who also appear in credit_record.csv

    cred_ids = cred_rd["ID"].unique()

    app_rd = app_rd[app_rd["ID"].isin(cred_ids)].reset_index(drop = True)

    # And viceversa

    app_ids = app_rd["ID"].unique()

    cred_rd = cred_rd[cred_rd["ID"].isin(app_ids)].reset_index(drop = True)

    ## Preprocessing: application_record

    # - FLAG_OWN_CAR, FLAG OWN REALTY: Y/N -> bool

    mapping = {"Y": 1, "N": 0}

    app_rd["FLAG_OWN_CAR"] = app_rd["FLAG_OWN_CAR"].map(mapping)
    app_rd["FLAG_OWN_REALTY"] = app_rd["FLAG_OWN_REALTY"].map(mapping)

    # - DAYS_BIRTH -> age (Count backwards from current day (0), -1 means yesterday)

    app_rd["AGE"] = app_rd["DAYS_BIRTH"]/-365

    # - DAYS_EMPLOYED -> months working (Count backwards from current day(0). If positive, it means the person currently unemployed.)

    # For some reason, unemployed people have 365243 in DAYS_UNEMPLOYED, and there are no zeroes
    app_rd.loc[app_rd["DAYS_EMPLOYED"] > 0, "DAYS_EMPLOYED"] = 0

    app_rd["YEARS_EMPLOYED"] = app_rd["DAYS_EMPLOYED"] / -365

    # - OCCUPATION_TYPE has nulls

    app_rd["OCCUPATION_TYPE"] = app_rd["OCCUPATION_TYPE"].fillna("None")

    # - NAME_INCOME_TYPE, NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS, NAME_HOUSING_TYPE, OCCUPATION_TYPE -> one-hot

    # will be handled in training as a pipeline

    app_rd.to_csv("./data/processed/application_record_processed.csv", index=False)

    cred_rd.to_csv("./data/processed/credit_record_processed.csv", index=False)

if __name__ == "__main__":
    preprocess_data()