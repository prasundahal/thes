import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def hex_to_rgb(hex_color):
    """
    Convert a hex color string (e.g., "#aabbcc") to an RGB tuple.
    Returns (np.nan, np.nan, np.nan) if the string is invalid or missing.
    """
    if pd.isnull(hex_color):
        return (np.nan, np.nan, np.nan)
    try:
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    except (ValueError, TypeError):
        return (np.nan, np.nan, np.nan)

def main():
    # Load your combined CSV file
    df = pd.read_csv("combined.csv")
    
    # Map genders to numeric values (Male=1, Female=0)
    gender_mapping = {'Male': 1, 'Female': 0}
    df['gender_num'] = df['gender'].map(gender_mapping)

    # Convert the dominant color hex to separate R, G, and B numerical columns
    rgb_values = df['dom_col_hex'].apply(lambda x: pd.Series(hex_to_rgb(x)))
    rgb_values.columns = ['color_R', 'color_G', 'color_B']
    df = pd.concat([df, rgb_values], axis=1)
    
    # Keep only the relevant numeric columns for correlation
    numeric_cols = ['price', 'qty_ordered', 'age', 'gender_num', 
                    'color_R', 'color_G', 'color_B']
    corr_df = df[numeric_cols]
    
    # Compute the correlation matrix
    corr_matrix = corr_df.corr()

    # Plot the correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap: Price, Quantity, Age, Gender & Color Channels")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
