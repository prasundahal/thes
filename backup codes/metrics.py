import pandas as pd

# Load data
recommendations_df = pd.read_csv('recommended_products.csv')
# recommendations_df = pd.read_csv('sorted_recommendations.csv')
original_df = pd.read_csv('pak_ecom_5000.csv')

# Preprocess data
recommendations_df['recommended'] = 1  # Add a column indicating recommendation
original_df['purchased'] = original_df['status'].apply(lambda x: 1 if x == 'complete' else 0)  # Convert 'complete' status to 1, others to 0

# Merge data based on common Customer ID
merged_df = pd.merge(recommendations_df, original_df, on='Customer ID', how='left')

# Calculate true positives, false positives, and false negatives
true_positives = merged_df[(merged_df['recommended'] == 1) & (merged_df['purchased'] == 1)].shape[0]
false_positives = merged_df[(merged_df['recommended'] == 1) & (merged_df['purchased'] == 0)].shape[0]
false_negatives = merged_df[(merged_df['recommended'] == 0) & (merged_df['purchased'] == 1)].shape[0]

# Calculate precision, recall, and F1 score
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Calculate accuracy
total_recommendations = merged_df[merged_df['recommended'] == 1].shape[0]
accuracy = true_positives / total_recommendations if total_recommendations > 0 else 0

# Print metrics
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
print("Accuracy:", accuracy)


# Load data
# recommendations_df = pd.read_csv('recommendations_collaborative.csv')
recommendations_df = pd.read_csv('combined.csv')
original_df = pd.read_csv('pak_ecom_5000.csv')

# Preprocess data
recommendations_df['recommended'] = 1  # Add a column indicating recommendation
original_df['purchased'] = original_df['status'].apply(lambda x: 1 if x == 'complete' else 0)  # Convert 'complete' status to 1, others to 0

# Merge data based on common Customer ID
merged_df = pd.merge(recommendations_df, original_df, on='Customer ID', how='left')

# Calculate true positives, false positives, and false negatives
true_positives = merged_df[(merged_df['recommended'] == 1) & (merged_df['purchased'] == 1)].shape[0]
false_positives = merged_df[(merged_df['recommended'] == 1) & (merged_df['purchased'] == 0)].shape[0]
false_negatives = merged_df[(merged_df['recommended'] == 0) & (merged_df['purchased'] == 1)].shape[0]

# Calculate precision, recall, and F1 score
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Calculate accuracy
total_recommendations = merged_df[merged_df['recommended'] == 1].shape[0]
accuracy = true_positives / total_recommendations if total_recommendations > 0 else 0

# Print metrics
print("")
print("Precision using colours :", precision)
print("Recall using colours :", recall)
print("F1 Score using colours :", f1_score)
print("Accuracy using colours :", accuracy)
