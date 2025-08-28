#%% To convert into appropriate UTF-8 format

import pandas as pd

try:
    # Attempt to read the file with default UTF-8 encoding to demonstrate the error
    df = pd.read_csv(r'nom_du_fichier_problematique.csv')
    print("File read successfully with default UTF-8 encoding.")
except UnicodeDecodeError as e:
    print(f"Caught expected error: {e}")
    # The error confirms the file is not UTF-8 encoded.
    # We will now read it with a more compatible encoding like 'latin-1' or 'cp1252'.
    # I'll use 'latin-1' as it's a common fallback.
    try:
        df = pd.read_csv(r'nom_du_fichier_problematique.csv', encoding='latin-1')
        print("File read successfully with latin-1 encoding.")

        # Save the file with explicit UTF-8 encoding to make it compatible with Label Studio
        output_filename = 'nom_du_nouveau_fichier_utf8.csv'
        df.to_csv(output_filename, index=False, encoding='utf-8')
        print(f"File successfully converted and saved as {output_filename}.")

    except Exception as e:
        print(f"An unexpected error occurred during re-encoding: {e}")

# Display the first few rows of the corrected DataFrame for inspection
print("\nFirst 5 rows of the converted DataFrame:")
print(df.head().to_markdown(index=False, numalign="left", stralign="left"))

# Display column info to verify the data types
print("\nDataFrame info:")
print(df.info())