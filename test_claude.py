import os
import pandas as pd


# Load the data from a CSV file
df = pd.read_csv("/home/zli/SafeDecoding/exaggerated-safety/xstest_v2_prompts.csv")

# Filter the rows where 'type' column values do not start with 'contrast_'
filtered_df = df[~df['type'].str.startswith('contrast_', na=False)]

# Extract the 'type' column from these filtered rows
non_contrast_types = filtered_df['id_v2'].tolist()



def parse_filename(filename):
    """
    Parse the filename to extract dataset, model, and defender names.
    
    :param filename: str - The filename to parse
    :return: tuple - (dataset, model, defender)
    """
    # Remove the file extension
    filename = filename.rstrip('.csv')
    
    # Split the filename by underscores
    parts = filename.split('_')
    
    # Check if the filename includes a dataset indicator like "completions"
    
    if 'xs' in parts:
        dataset = 'xs'
        model = parts[1]
        defender = '_'.join(parts[2:])
    elif 'oa' in parts:
        dataset = 'oa'
        model = parts[1]
        defender = '_'.join(parts[2:])
    elif 'law' in parts:
        dataset = 'law'
        model = parts[1]
        defender = '_'.join(parts[2:])
    elif 'math' in parts:
        dataset = 'math'
        model = parts[1]
        defender = '_'.join(parts[2:])
    elif 'health' in parts:
        dataset = 'health'
        model = parts[1]
        defender = '_'.join(parts[2:])  
    else:
        print("no dataset")
    return dataset, model, defender



def browse_and_process_files(root_path,out_name):
    # Initialize counters for each class
    
    data_for_csv = []
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            
            # Check if the file is a CSV file
            if filename.endswith(".csv"):
                class_counters = {
        'full_compliance': 0,
        'full_refusal': 0,
        'partial_refusal': 0
            }
    
    # Class map for easy comparison
                class_map = {
        '1_full_compliance': 'full_compliance',
        '2_full_refusal': 'full_refusal',
        '3_partial_refusal': 'partial_refusal'
            }
                file_path = os.path.join(dirpath, filename)
                tot_cnt=0
                tot_latency=0
                tot_input=0
                tot_output=0
                tot_en=0
            try:
                    # Read CSV file to check if it contains the required columns
                    df = pd.read_csv(file_path)
                        
                            
                    # Check for the class and update the corresponding counter
                    
                    
                    for index, row in df.iterrows():
                        dataset, model, defender=parse_filename(filename)
                        if dataset=='xs' and row['id'] in non_contrast_types:
                            pass
                        else:
                            continue
                        row['class'] = row['class'].lower()  # Normalize class names
                        matched_classes = [key for label, key in class_map.items() if key in row['class']]
                        
                        # Only update counts if exactly one class is present in the file
                        if len(matched_classes) == 1:
                            class_counters[matched_classes[0]] += 1
                            tot_cnt+=1
                            tot_latency+=row['latency']
                            tot_input+=row['input tokens']
                            tot_output+=row['output tokens']
                            tot_en+=row['energy']
                    
                    entry = {
                        "dataset": dataset,
                        "model": model,
                        "defender": defender,
                        "latency": f"{tot_latency/tot_cnt:.2f}",
                        "input tokens": f"{tot_input/tot_cnt:.2f}",
                        "output tokens": f"{tot_output/tot_cnt:.2f}",
                        "energy": f"{tot_en/tot_cnt:.2f}",
                    }
                    
                    for classname, count in class_counters.items():
                        if tot_cnt > 0:
                            percentage = (count / tot_cnt) * 100
                            entry[classname]=f"{percentage:.2f}"
                            print(f"{classname}: {count} ({percentage:.2f}%)")
                    data_for_csv.append(entry)
                        

                

                    
            except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
            df = pd.DataFrame(data_for_csv)
    
            # Save to CSV
            df.to_csv(out_name, index=False)
                   

    

# Example usage:
root_directory = "/home/zli/SafeDecoding/exaggerated-safety/results/analysis"
browse_and_process_files(root_directory,"analysis-xs.csv")

