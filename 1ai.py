import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# Load the data
ecom_data = pd.read_csv('pak_ecom_5000.csv')
customer_data = pd.read_csv('customer_info.csv')

# Merge the datasets on Customer ID
merged_data = pd.merge(ecom_data, customer_data, on='Customer ID')

# Create a pivot table for collaborative filtering
pivot_table = merged_data.pivot_table(index='Customer ID', columns='item_id', values='qty_ordered', fill_value=0)

# Normalize interaction scores
pivot_table_normalized = pivot_table.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)

# Optimize Truncated SVD for dimensionality reduction
svd = TruncatedSVD(n_components=100)
matrix = svd.fit_transform(pivot_table_normalized)
explained_variance = np.cumsum(svd.explained_variance_ratio_)
optimal_components = np.argmax(explained_variance >= 0.9) + 1  # 90% variance
svd = TruncatedSVD(n_components=optimal_components)
matrix = svd.fit_transform(pivot_table_normalized)

# Map Customer ID to pivot_table indices
customer_id_to_index = {customer_id: idx for idx, customer_id in enumerate(pivot_table.index)}

# Function to recommend products based on age and gender
def recommend_products(age, gender):
    similar_customers = merged_data[(merged_data['Age'] == age) & (merged_data['Gender'] == gender)]['Customer ID'].unique()
    similar_indices = [customer_id_to_index[cust_id] for cust_id in similar_customers if cust_id in customer_id_to_index]

    # Correct weights and calculate new customer vector
    weights = pivot_table_normalized.sum(axis=1).values  # Sum of interaction scores
    weights /= weights.sum()  # Normalize weights
    new_customer_vector = np.average(matrix, axis=0, weights=weights)

    similarities = cosine_similarity([new_customer_vector], matrix[similar_indices])
    most_similar_indices = [similar_indices[i] for i in similarities[0].argsort()[::-1]]
    similar_customer_ids = [pivot_table.index[idx] for idx in most_similar_indices]
    recommendations = merged_data[merged_data['Customer ID'].isin(similar_customer_ids)]

    recommended_products = recommendations[['item_id', 'sku', 'category_name_1', 'price', 'qty_ordered', 'Year', 'Month', 'Customer ID']]
    recommended_products.to_csv('recommended_products.csv', index=False)

    return recommended_products, similar_customer_ids, similarities

# Example usage
age = 30
gender = 'Male'
recommendations, similar_customer_ids, similarities = recommend_products(age, gender)

# Evaluation and Plots

# 1. Precision-Recall Curve
y_true = np.random.randint(0, 2, size=len(similarities[0]))  # Replace with actual relevant/non-relevant labels
y_scores = similarities[0]  # Predicted similarity scores

precision, recall, _ = precision_recall_curve(y_true, y_scores)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# 2. ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# 3. Top-N Accuracy Plot
def top_n_accuracy(recommendations, relevant_items, N):
    hits = 0
    for rec, rel in zip(recommendations, relevant_items):
        rec = [rec] if not isinstance(rec, (list, np.ndarray)) else rec
        rel = [rel] if not isinstance(rel, (list, np.ndarray)) else rel
        if any(item in rel for item in rec[:N]):
            hits += 1
    return hits / len(recommendations)

N_values = np.arange(1, 21)
accuracies = [top_n_accuracy(recommendations['item_id'].tolist(), y_true, N) for N in N_values]

plt.figure(figsize=(8, 6))
plt.plot(N_values, accuracies, marker='o')
plt.xlabel('Top-N')
plt.ylabel('Accuracy')
plt.title('Top-N Accuracy Plot')
plt.show()

# 4. Heatmap of User-Item Interaction Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table_normalized, cmap="YlGnBu", cbar=False)
plt.title('User-Item Interaction Matrix')
plt.xlabel('Item ID')
plt.ylabel('Customer ID')
plt.show()

# 5. Distribution of Predicted Scores
plt.figure(figsize=(8, 6))
plt.hist(y_scores, bins=50, color='skyblue', edgecolor='black')
plt.xlabel('Predicted Scores')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Scores')
plt.show()

# 6. Confusion Matrix
y_pred = (y_scores >= 0.5).astype(int)  # Example threshold
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 7. Histogram of Recommendation Counts
plt.figure(figsize=(8, 6))
plt.hist(recommendations['item_id'], bins=30, color='orange', edgecolor='black')
plt.xlabel('Item ID')
plt.ylabel('Frequency')
plt.title('Histogram of Recommendation Counts')
plt.show()
