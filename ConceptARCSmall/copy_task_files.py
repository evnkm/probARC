import os
import shutil

# Define the source and destination directories
source_base_dir = '/om/user/evan_kim/966/BARC/ConceptARC'
dest_base_dir = '/om/user/evan_kim/966/BARC/ConceptARCSmall/taskOrder1'

# Task categories and their corresponding files
task_categories_order = {
    "AboveBelow": [
        'AboveBelow7.json',
        'AboveBelow3.json',
        'AboveBelow6.json',
        'AboveBelow8.json',
        'AboveBelow2.json'
    ],
    "Center": [
        'Center3.json',
        'Center6.json',
        'Center5.json',
        'Center4.json',
        'Center7.json'
    ],
    "ExtendToBoundary": [
        'ExtendToBoundary3.json',
        'ExtendToBoundary4.json',
        'ExtendToBoundary2.json',
        'ExtendToBoundary9.json',
        'ExtendToBoundary6.json'
    ],
    "InsideOutside": [
        'InsideOutside5.json',
        'InsideOutside1.json',
        'InsideOutside3.json',
        'InsideOutside7.json',
        'InsideOutside2.json'
    ],
    "SameDifferent": [
        'SameDifferent3.json',
        'SameDifferent1.json',
        'SameDifferent7.json',
        'SameDifferent6.json',
        'SameDifferent9.json'
    ]
}

# Ensure destination directory exists
os.makedirs(dest_base_dir, exist_ok=True)

# Copy files
for category, files in task_categories_order.items():
    category_source_dir = os.path.join(source_base_dir, category)
    category_dest_dir = os.path.join(dest_base_dir, category)
    
    # Create category subdirectory in destination
    os.makedirs(category_dest_dir, exist_ok=True)
    
    # Copy each file
    for file in files:
        source_file = os.path.join(category_source_dir, file)
        dest_file = os.path.join(category_dest_dir, file)
        
        if os.path.exists(source_file):
            shutil.copy2(source_file, dest_file)
            print(f"Copied {source_file} to {dest_file}")
        else:
            print(f"Warning: Source file {source_file} does not exist")

print("File copying completed.")
