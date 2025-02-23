import pandas as pd

# Load data for without colors
recommendations_no_color_df = pd.read_csv('recommended_products.csv')
original_df = pd.read_csv('pak_ecom_5000.csv')

# Load data for with colors
recommendations_with_color_df = pd.read_csv('combined.csv')

# Preprocess original data
original_df['purchased'] = original_df['status'].apply(lambda x: 1 if x == 'complete' else 0)  # Convert 'complete' status to 1, others to 0

# Function to calculate metrics
def calculate_metrics(recommendations_df, original_df):
    # Add recommended column
    recommendations_df['recommended'] = 1

    # Merge data
    merged_df = pd.merge(recommendations_df, original_df, on='Customer ID', how='right')
    merged_df['recommended'] = merged_df['recommended'].fillna(0)  # Treat non-recommended products as 0

    # Calculate true positives, false positives, true negatives, false negatives
    true_positives = merged_df[(merged_df['recommended'] == 1) & (merged_df['purchased'] == 1)].shape[0]
    false_positives = merged_df[(merged_df['recommended'] == 1) & (merged_df['purchased'] == 0)].shape[0]
    true_negatives = merged_df[(merged_df['recommended'] == 0) & (merged_df['purchased'] == 0)].shape[0]
    false_negatives = merged_df[(merged_df['recommended'] == 0) & (merged_df['purchased'] == 1)].shape[0]

    # Calculate precision, recall, F1 score, and accuracy
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / merged_df.shape[0] if merged_df.shape[0] > 0 else 0

    return precision, recall, f1_score, accuracy

# Calculate metrics for without colors
precision_no_color, recall_no_color, f1_score_no_color, accuracy_no_color = calculate_metrics(recommendations_no_color_df, original_df)

# Calculate metrics for with colors
precision_with_color, recall_with_color, f1_score_with_color, accuracy_with_color = calculate_metrics(recommendations_with_color_df, original_df)

# Print results
print("Metrics Without Colors:")
print(f"Precision: {precision_no_color:.4f}")
print(f"Recall: {recall_no_color:.4f}")
print(f"F1 Score: {f1_score_no_color:.4f}")
print(f"Accuracy: {accuracy_no_color:.4f}")
print()

print("Metrics With Colors:")
print(f"Precision: {precision_with_color:.4f}")
print(f"Recall: {recall_with_color:.4f}")
print(f"F1 Score: {f1_score_with_color:.4f}")
print(f"Accuracy: {accuracy_with_color:.4f}")