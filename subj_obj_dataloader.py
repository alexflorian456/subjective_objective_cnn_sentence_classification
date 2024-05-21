import os
import re

# Define the source directory
source_directory = '/mnt/c/Users/flale/Downloads/Sub-Obj'

# Define the output files
subj_output_file = '/home/alex/nlp/cnn-text-classification-tf/data/subj-obj/all_subj.txt'
obj_output_file = '/home/alex/nlp/cnn-text-classification-tf/data/subj-obj/all_obj.txt'

# Open output files in write mode
with open(subj_output_file, 'w') as subj_file, open(obj_output_file, 'w') as obj_file:
    # Walk through the source directory
    for root, dirs, files in os.walk(source_directory):
        for file in files:
            file_path = os.path.join(root, file)
            # Check if the file name contains "subj"
            if 'subj' in file.lower():
                with open(file_path, 'r') as f:
                    subj_file.write(f.read() + '\n')  # Append file content to the subj output file
            # Check if the file name contains "obj"
            elif 'obj' in file.lower():
                with open(file_path, 'r') as f:
                    obj_file.write(f.read() + '\n')  # Append file content to the obj output file

print("Contents have been successfully combined into respective files.")

input_files = [subj_output_file, obj_output_file]

import re

# Regular expression to match lines starting with numbering
numbering_pattern = re.compile(r'^\s*\d+\.\s*')

for input_file in input_files:
    # Read the contents of the file
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Process lines: remove empty lines and numbering
    cleaned_lines = []
    for line in lines:
        # Remove empty lines
        if line.strip():
            # Remove numbering at the start of the line
            cleaned_line = numbering_pattern.sub('', line)
            cleaned_lines.append(cleaned_line)

    # Write the cleaned content back to the file (or to a new file)
    with open(input_file, 'w') as file:
        file.writelines(cleaned_lines)

print("Files have been cleaned successfully.")

