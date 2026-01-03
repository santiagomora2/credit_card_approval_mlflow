import pandas as pd

def extract_features_and_label():
    app_rd = pd.read_csv("./data/processed/application_record_processed.csv")
    cred_rd = pd.read_csv("./data/processed/credit_record_processed.csv")

    ## application_record features

    app_rd.drop(columns = ["CNT_CHILDREN", "FLAG_MOBIL", "DAYS_BIRTH", "DAYS_EMPLOYED"], inplace=True)

    ## credit_record labels

    def classify_client(statuses):
        """
        Classify a client based on their STATUS history
        
        Rules:
        1. EXCLUDE if < 3 values different from 'X' (not enough data)
        2. BAD (target = 0) if STATUS ≥ 3 at any point (3, 4, 5)
        3. GOOD (target = 1) if all values in (X, C, 0, 1, 2) only
        """
        # Remove 'X' values for counting data points
        non_x_values = [s for s in statuses if s != 'X']
        
        # Rule 1: Exclude if < 3 non-X values
        if len(non_x_values) < 3:
            return None
        
        # Rule 2: BAD if any STATUS ≥ 3 (these are strings, so compare accordingly)
        # Convert numeric strings to integers for comparison
        for status in statuses:
            if status in ['3', '4', '5']:
                return 0
        
        # Rule 3: GOOD if all values in allowed set
        allowed_statuses = {'X', 'C', '0', '1', '2'}
        if all(status in allowed_statuses for status in statuses):
            return 1
        
        # If none of the above, exclude (safety fallback)
        return None

    # Get STATUS history for each ID
    status_by_client = cred_rd.groupby('ID')['STATUS'].apply(list)

    # Classify each client
    client_labels = status_by_client.apply(classify_client)

    # Convert to DataFrame
    client_labels_df = client_labels.reset_index()
    client_labels_df.columns = ['ID', 'TARGET']

    # Merge on ID

    app_rd_labeled = app_rd.merge(
        client_labels_df[client_labels_df['TARGET'].notna()], 
        on='ID', 
        how='inner'
    )

    ## Drop duplicates and resolve conflicts

    feature_cols = [col for col in app_rd_labeled.columns if col not in ['ID', 'TARGET']]
        
    # Drop duplicates, keep first occurrence. Handle conflicts: take the worst case (BAD if any BAD exists)

    app_rd_deduped = app_rd_labeled.sort_values('TARGET', ascending=True)\
                                    .drop_duplicates(subset=feature_cols, keep='first')\
                                    .reset_index(drop=True)
    app_rd_deduped.to_csv("./data/features/data.csv", index=False)

if __name__ == "__main__":
    extract_features_and_label()