import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(input_file, output_file):
    # Load dataset in chunks to handle large files
    chunks = pd.read_csv(input_file, chunksize=10000, low_memory=False)
    processed_chunks = []

    for chunk in chunks:
        # Drop irrelevant columns
        chunk.drop(columns=["Flow ID", "Source IP", "Destination IP", "Timestamp"], errors='ignore', inplace=True)
        
        # Handle missing values
        chunk.dropna(inplace=True)

        # Dynamically detect the label column
        label_column = [col for col in chunk.columns if "label" in col.lower() or "class" in col.lower()]
        if not label_column:
            raise KeyError("No label column found in the dataset. Please check the dataset structure.")
        label_column = label_column[0]

        # Encode labels (BENIGN = 0, ATTACK = 1)
        label_encoder = LabelEncoder()
        chunk[label_column] = label_encoder.fit_transform(chunk[label_column])

        # Rename label column to a standard name
        chunk.rename(columns={label_column: "Label"}, inplace=True)

        # Replace inf/-inf values with NaN and then drop them
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk.dropna(inplace=True)

        # Normalize numeric features
        scaler = StandardScaler()
        numeric_columns = chunk.drop(columns=["Label"]).columns

        # Ensure numeric features have valid finite values
        for col in numeric_columns:
            if not np.isfinite(chunk[col]).all():
                print(f"Warning: Column '{col}' contains invalid values and will be removed.")
                chunk.drop(columns=[col], inplace=True)

        # Apply normalization
        chunk[numeric_columns] = scaler.fit_transform(chunk[numeric_columns])

        processed_chunks.append(chunk)

    # Combine all processed chunks
    processed_data = pd.concat(processed_chunks, ignore_index=True)
    processed_data.to_csv(output_file, index=False)
    print("Data preprocessing complete. Processed data saved to:", output_file)

if __name__ == "__main__":
    preprocess_data("data/Tuesday-WorkingHours.csv", "data/processed_data.csv")
