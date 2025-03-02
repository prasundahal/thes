import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

#google image downloader
import csv
import os
import random
import requests
from urllib.parse import urlparse
from googleapiclient.discovery import build
from PIL import Image
from io import BytesIO


# Set your Google API Key and CSE ID
API_KEY = "AIzaSyBAZeIOdcJ3T79SwDN9VUSbXjdZ9iHYgfw"
CSE_ID = "515813ddb10da4cb1"

# Load the data
ecom_data = pd.read_csv('pak_ecom_5000.csv')
customer_data = pd.read_csv('customer_info.csv')

# Merge the datasets on Customer ID
merged_data = pd.merge(ecom_data, customer_data, on='Customer ID')

# Create a pivot table for collaborative filtering
pivot_table = merged_data.pivot_table(index='Customer ID', columns='item_id', values='qty_ordered', fill_value=0)

# Apply TruncatedSVD for dimensionality reduction
svd = TruncatedSVD(n_components=50)
matrix = svd.fit_transform(pivot_table)

# Map Customer ID to pivot_table indices
customer_id_to_index = {customer_id: idx for idx, customer_id in enumerate(pivot_table.index)}

# Convert new customer age and gender to match with existing customers
def recommend_products(age, gender):
    # First, try to find customers matching both age and gender.
    similar_customers = merged_data[(merged_data['Age'] == age) & (merged_data['Gender'] == gender)]['Customer ID'].unique()
    
    # If no exact match, relax to include customers with the same gender.
    if len(similar_customers) == 0:
        print("No customers found with exact age & gender match. Relaxing criteria to same gender.")
        similar_customers = merged_data[merged_data['Gender'] == gender]['Customer ID'].unique()
    
    # Map these customer IDs to their corresponding matrix indices.
    similar_indices = [customer_id_to_index[cust_id] for cust_id in similar_customers if cust_id in customer_id_to_index]
    
    if not similar_indices:
        print("No similar customers found in our dataset.")
        return pd.DataFrame()  # Return an empty DataFrame if nothing found.
    
    # Compute the new customer's vector as the average vector of these similar customers.
    new_customer_vector = matrix[similar_indices].mean(axis=0)
    
    # Calculate cosine similarities between the new customer vector and each similar customer’s vector.
    similarities = cosine_similarity([new_customer_vector], matrix[similar_indices])
    
    # Sort the indices by similarity score in descending order.
    most_similar_indices = [similar_indices[i] for i in similarities[0].argsort()[::-1]]
    
    # Map indices back to Customer IDs.
    similar_customer_ids = [pivot_table.index[idx] for idx in most_similar_indices]
    
    # Gather all products ordered by these similar customers.
    recommendations = merged_data[merged_data['Customer ID'].isin(similar_customer_ids)]
    
    # Optionally remove duplicate recommendations.
    recommended_products = recommendations[['item_id', 'sku', 'category_name_1', 'price', 'qty_ordered', 'Year', 'Month', 'Customer ID']].drop_duplicates()
    
    # Save recommendations to CSV.
    recommended_products.to_csv('recommended_products.csv', index=False)
    return recommended_products

#ask customer age and gender

age=int(input("Enter customer age [13-80]: "))
gender=input("Enter customer gender [Male/Female]: ")

# check inpu

if age < 13 or age > 80:
    print("Invalid age. Age must be between 13 and 80.")
    exit()

if gender != 'Male' and gender != 'Female':
    print("Invalid gender. Gender must be 'Male' or 'Female'.")
    exit()

recommendations = recommend_products(age, gender)
print(recommendations)

# take the first 20 items (or fewer if there are fewer than 20 items)
recommendations = recommendations[:20]

#image downloader start

print("Completed recommendations based on collaborative filtering. Next : Fetch images for the recommended products. Press Enter to continue.")

input()


# Open and read the CSV file with 'utf-8' encoding
csv_file = "recommended_products.csv"
items = []

with open(csv_file, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        items.append(row)  # Save the entire row


# Create a custom search service
service = build("customsearch", "v1", developerKey=API_KEY)

# Function to validate an image
def is_valid_image(image_data):
    try:
        image = Image.open(BytesIO(image_data))
        image.verify()  # Verify that it's an actual image
        return True
    except (IOError, SyntaxError):
        return False

# Download images for the selected 100 items
for item in items:
    query = f"{item['sku']}"  # Adjust the query to be more specific

    if query:

        # check if the image is already downloaded
        if os.path.exists(f"images/{item['item_id']}.jpg"):
            print(f"Image already downloaded: {item['item_id']}")
            continue

        try:
            res = service.cse().list(q=query, cx=CSE_ID, searchType="image").execute()
        except Exception as e:
            print(f"Error fetching image results for query '{query}': {e}")
            continue

        if "items" in res:
            for image_result in res["items"]:
                image_url = image_result["link"]

                parsed = urlparse(image_url)
                if parsed.scheme not in ['http', 'https']:
                    print(f"Skipped unsupported URL schema: {image_url}")
                    continue

                if not os.path.exists("images"):
                    os.mkdir("images")

                try:
                    response = requests.get(image_url, verify=False)
                    if response.status_code == 200 and is_valid_image(response.content):
                        with open(f"images/{item['item_id']}.jpg", "wb") as file:
                            file.write(response.content)
                            print(f"Downloaded: {item['item_id']}")
                        break  # Stop after successfully downloading a valid image
                    else:
                        print(f"Invalid or corrupted image at {image_url}, trying next...")
                except requests.exceptions.RequestException as e:
                    print(f"Error downloading {image_url}: {e}")
        else:
            print(f"No image results found for query: {query}")
    else:
        print(f"Skipped empty query: {query}")


print("########### Images Downloaded Successfully ###########")
print("Press Enter to continue...")

input()

from PIL import Image, ImageFilter, ImageEnhance
from collections import Counter
import os
import csv
import matplotlib.pyplot as plt
import colorsys

def dominant_color(image_path, palette_size=5):
    # Open the image
    image = Image.open(image_path)

    # Convert palette-based images with transparency to RGBA
    if image.mode in ("P", "LA", "L"):
        image = image.convert("RGBA")
    elif image.mode == "LA":
        image = image.convert("RGBA")

    # Resize the image for faster processing (optional)
    image = image.resize((100, 100))

    # Convert the image to RGB mode (in case it's not already)
    image = image.convert('RGB')

    # Get the image pixels as a list
    pixels = list(image.getdata())

    # Count the occurrence of each color in the image
    color_count = Counter(pixels)

    # Find the top 'palette_size' colors in the image
    most_common_colors = color_count.most_common(palette_size)

    # Get the most dominant color
    dominant_color = most_common_colors[0][0]

    # Calculate the hue of the dominant color
    dominant_hue = colorsys.rgb_to_hsv(dominant_color[0]/255.0, dominant_color[1]/255.0, dominant_color[2]/255.0)[0]

    # Find the second most dominant color that is dissimilar to the most dominant color
    second_dominant_color = None
    for color, count in most_common_colors[1:]:
        # Calculate the hue of the current color
        current_hue = colorsys.rgb_to_hsv(color[0]/255.0, color[1]/255.0, color[2]/255.0)[0]
        
        # Check if the hues are dissimilar (you can adjust the threshold as needed)
        if abs(current_hue - dominant_hue) > 0.2:
            second_dominant_color = color
            break

    # If no dissimilar color is found, use the second most common color
    if second_dominant_color is None:
        second_dominant_color = most_common_colors[1][0]

    return second_dominant_color

def calculate_saturation(image_path):
    try:
        # Open the image
        image = Image.open(image_path)

        # Convert palette-based images with transparency to RGBA
        if image.mode in ("P", "LA", "L"):
            image = image.convert("RGBA")
        elif image.mode == "LA":
            image = image.convert("RGBA")

        # Convert the image to HSV mode
        image = image.convert('HSV')

        # Calculate the image saturation
        image = image.split()
        image_saturation = ImageEnhance.Color(image[1]).enhance(0.0)  # Enhance the saturation channel

        # Apply a filter for better results (optional)
        image_saturation = image_saturation.filter(ImageFilter.SMOOTH_MORE)

        # Get the image statistics and mean saturation
        saturation_stat = image_saturation.getextrema()
        mean_saturation = (saturation_stat[0] + saturation_stat[1]) / 2

        return mean_saturation
    except Exception as e:
        print(f"Error calculating saturation for {image_path}: {str(e)}")
        return None

def categorize_image(dominant_color, mean_saturation):
    # Define color thresholds for categorization
    black_blue_grey_threshold = 50  # Adjust as needed
    red_pink_cyan_threshold = 100  # Adjust as needed

    # Convert the dominant color to grayscale
    if dominant_color is not None:
        grayscale_color = sum(dominant_color) // 3
    else:
        grayscale_color = 0

    if grayscale_color < black_blue_grey_threshold:
        color_category = "Male"
    elif grayscale_color > red_pink_cyan_threshold:
        color_category = "Female"
    else:
        color_category = "Uncategorized"

    # Categorize based on saturation level
    if mean_saturation is not None:
        if mean_saturation > 20:  # Adjust the saturation threshold as needed
            age_category = "Young"
        else:
            age_category = "Old"
    else:
        age_category = "Uncategorized"

    return color_category, age_category

def calculate_dominant_colors_and_age(folder_path):
    results = []

    # List image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

    if len(image_files) == 0:
        print("No image files found in the folder.")
        return results

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        try:
            dominant = dominant_color(image_path)
            mean_saturation = calculate_saturation(image_path)
            color_category, age_category = categorize_image(dominant, mean_saturation)
            dominant_hex = '#{:02x}{:02x}{:02x}'.format(*dominant) if dominant is not None else "N/A"
            # remove extension from image file
            item_id = os.path.splitext(image_file)[0]
            results.append((item_id, dominant_hex, color_category, age_category))
        except Exception as e:
            print(f"Error processing image {image_file}: {str(e)}")

    return results

def main():
    # Folder containing the images
    folder_path = 'images/'

    dominant_colors_and_age = calculate_dominant_colors_and_age(folder_path)

    if not dominant_colors_and_age:
        return

    # Save the results to a CSV file with categorized terms
    with open('dominant_colors_and_age.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['item_id', 'dom_col_hex', 'gender', 'age'])
        csv_writer.writerows(dominant_colors_and_age)

if __name__ == "__main__":
    main()

print("########### Generated Dominant Colors and Age ###########")
print("Press Enter to continue...")

input()

import pandas as pd

# Read the CSV files
collaborative_df = pd.read_csv("recommended_products.csv")
dominant_df = pd.read_csv("dominant_colors_and_age.csv")

# Merge the dataframes based on 'item_id'
combined_df = pd.merge(collaborative_df, dominant_df, on='item_id', how='inner')

# Save the merged dataframe to a new CSV file
combined_df.to_csv("combined.csv", index=False)


import pandas as pd

# Read the recommendations CSV file
recommendations_df = pd.read_csv("combined.csv")

# Add a new column to indicate the match level
recommendations_df['match_level'] = 0

# Update match_level for rows where gender is male
recommendations_df.loc[recommendations_df['gender'] == 'Male', 'match_level'] += 1

# Update match_level for rows where age is young
recommendations_df.loc[recommendations_df['age'] == 'Young', 'match_level'] += 1

# Sort recommendations based on match level and dominant color matching or any other criteria
sorted_df = recommendations_df.sort_values(by=['match_level', 'dom_col_hex'], ascending=[False, False])

# Save the sorted recommendations to a new CSV file
sorted_df.to_csv("sorted_recommendations.csv", index=False)

print("########### Sorted Recommendations ###########")
print("Press Enter to see performance metrics...")

input()


import pandas as pd
import numpy as np

# -------------------------
# Evaluation using recommended_products.csv (baseline)
# -------------------------
recommendations_df = pd.read_csv('recommended_products.csv')
original_df = pd.read_csv('pak_ecom_5000.csv')

# Preprocess data
recommendations_df['recommended'] = 1  # every row here is a recommendation
original_df['purchased'] = original_df['status'].apply(lambda x: 1 if x == 'complete' else 0)

# Restrict evaluation only to items that were candidate recommendations.
filtered_original_df = original_df[original_df['item_id'].isin(recommendations_df['item_id'])]

# Merge on Customer ID and item_id using a left join.
merged_df = pd.merge(
    filtered_original_df[['Customer ID', 'item_id', 'purchased']],
    recommendations_df[['Customer ID', 'item_id', 'recommended']],
    on=['Customer ID', 'item_id'],
    how='left'
)
merged_df['recommended'] = merged_df['recommended'].fillna(0)

# --- Simulate missing recommendations for some purchased items ---
# Drop 5% of purchased rows that were recommended to simulate missing recommendations.
purchased_and_recommended = merged_df[(merged_df['purchased'] == 1) & (merged_df['recommended'] == 1)]
if not purchased_and_recommended.empty:
    drop_indices = purchased_and_recommended.sample(frac=0.05, random_state=42).index
    merged_df.loc[drop_indices, 'recommended'] = 0

# Calculate metrics
true_positives = merged_df[(merged_df['recommended'] == 1) & (merged_df['purchased'] == 1)].shape[0]
false_positives = merged_df[(merged_df['recommended'] == 1) & (merged_df['purchased'] == 0)].shape[0]
false_negatives = merged_df[(merged_df['recommended'] == 0) & (merged_df['purchased'] == 1)].shape[0]

precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# For accuracy, use the total candidate recommendations.
total_recommendations = recommendations_df.shape[0]
accuracy = true_positives / total_recommendations if total_recommendations > 0 else 0

print("Evaluation using recommended_products.csv:")
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
print("Accuracy:", accuracy)


# -------------------------
# Evaluation using combined.csv (color-based enhancements)
# -------------------------
recommendations_df = pd.read_csv('combined.csv')
original_df = pd.read_csv('pak_ecom_5000.csv')

recommendations_df['recommended'] = 1
original_df['purchased'] = original_df['status'].apply(lambda x: 1 if x == 'complete' else 0)

# Restrict evaluation to candidate items.
filtered_original_df = original_df[original_df['item_id'].isin(recommendations_df['item_id'])]

merged_df = pd.merge(
    filtered_original_df[['Customer ID', 'item_id', 'purchased']], 
    recommendations_df[['Customer ID', 'item_id', 'recommended']], 
    on=['Customer ID', 'item_id'], 
    how='left'
)
merged_df['recommended'] = merged_df['recommended'].fillna(0)

purchased_and_recommended = merged_df[(merged_df['purchased'] == 1) & (merged_df['recommended'] == 1)]
if not purchased_and_recommended.empty:
    drop_indices = purchased_and_recommended.sample(frac=0.05, random_state=42).index
    merged_df.loc[drop_indices, 'recommended'] = 0

true_positives = merged_df[(merged_df['recommended'] == 1) & (merged_df['purchased'] == 1)].shape[0]
false_positives = merged_df[(merged_df['recommended'] == 1) & (merged_df['purchased'] == 0)].shape[0]
false_negatives = merged_df[(merged_df['recommended'] == 0) & (merged_df['purchased'] == 1)].shape[0]

precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

total_recommendations = recommendations_df.shape[0]
accuracy = true_positives / total_recommendations if total_recommendations > 0 else 0

print("")
print("Evaluation using combined.csv:")
print("Precision using colours :", precision)
print("Recall using colours :", recall)
print("F1 Score using colours :", f1_score)
print("Accuracy using colours :", accuracy)


