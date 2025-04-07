import pandas as pd
import re
import os
import sys

def convert_txt_to_csv(input_file, output_file):
    """
    Converts a text file with barcode counts to a CSV file.
    
    Args:
        input_file (str): Path to the input text file
        output_file (str): Path to the output CSV file
    """
    print(f"Reading file: {input_file}")
    
    # Read the entire file content
    with open(input_file, 'r') as file:
        content = file.read()
    
    # Find where the column headers end and the data begins
    headers_match = re.search(r'(TimePoint\s+Condition\s+Replicate\s+[^\n]+)', content)
    
    if not headers_match:
        print("Error: Could not find headers. Check file format.")
        return
    
    header_line = headers_match.group(1)
    
    # Split the header line to get column names
    # Handle both space and tab delimiters
    if '\t' in header_line:
        headers = header_line.split('\t')
    else:
        headers = header_line.split()
    
    print(f"Found {len(headers)} column headers")
    
    # Read the file line by line to extract data rows
    with open(input_file, 'r') as file:
        # Skip the header line
        line = file.readline()
        
        # Initialize list to store data rows
        data_rows = []
        
        # Read each subsequent line
        for line in file:
            line = line.strip()
            if line:  # Skip empty lines
                if '\t' in line:
                    values = line.split('\t')
                else:
                    values = line.split()
                
                # Only process lines that look like data (start with a number or letter)
                if values and (values[0].isdigit() or values[0].isalpha()):
                    # Handle case where there might be more values than headers
                    if len(values) > len(headers):
                        values = values[:len(headers)]
                    
                    # Handle case where there might be fewer values than headers
                    while len(values) < len(headers):
                        values.append("")
                    
                    data_rows.append(values)
    
    print(f"Found {len(data_rows)} data rows")
    
    # Create DataFrame
    df = pd.DataFrame(data_rows, columns=headers)
    
    # Convert numeric columns to appropriate types
    for col in df.columns[3:]:  # Skip TimePoint, Condition, Replicate columns
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            # Keep as is if conversion fails
            print(f"Could not convert column '{col}' to numeric. Keeping as is.")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to CSV
    print(f"Saving to CSV file: {output_file}")
    df.to_csv(output_file, index=False)
    print("Conversion complete!")

if __name__ == "__main__":
    # Define the project base directory
    # Adjust this to point to your PyFitSeq directory
    base_dir = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq"
    
    # Define the input file path
    input_file = os.path.join(base_dir, "farah_data/farah_raw_data/Filtered_Counts.txt")
    
    # Define the output file path
    output_dir = os.path.join(base_dir, "farah_data/outputs")
    output_file = os.path.join(output_dir, "barcode_counts.csv")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    # Run the conversion
    convert_txt_to_csv(input_file, output_file)
