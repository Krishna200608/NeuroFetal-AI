import json
import os

nb_path = r"d:\Research Project\Research_Project\Code\notebooks\Training_Colab.ipynb"

if not os.path.exists(nb_path):
    print(f"Error: File not found at {nb_path}")
    exit(1)

print(f"Reading {nb_path}...")
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

original_cell_count = len(nb['cells'])
new_cells = []
training_loop_seen = False

print("Iterating through cells...")
for i, cell in enumerate(nb['cells']):
    # Clean outputs for ALL cells
    if 'outputs' in cell:
        cell['outputs'] = []
    if 'execution_count' in cell:
        cell['execution_count'] = None

    source_list = cell.get('source', [])
    source = "".join(source_list)
    
    # Identify the new training loop cell
    if "Training Loop with Git Push" in source:
        print(f"Found Training Loop at cell index {i}")
        training_loop_seen = True
        new_cells.append(cell)
        continue

    # Logic for cells AFTER the training loop
    if training_loop_seen:
        is_redundant = False
        
        # Redundant Git operations
        if "!git config" in source and "user.email" in source: 
            print(f"Removing redundant git config at index {i}")
            is_redundant = True
        elif "!git commit" in source and "Updates from the colab" in source:
            print(f"Removing redundant git commit at index {i}")
            is_redundant = True
        elif "!git push origin main" in source and len(source.strip().split('\n')) <= 2:
            # Only remove if it's a standalone push, not part of a larger script (though the loop has its own push)
            print(f"Removing redundant git push at index {i}")
            is_redundant = True
        elif "!git add ." in source and len(source.strip().split('\n')) <= 2:
            print(f"Removing redundant git add at index {i}")
            is_redundant = True
        elif "!git pull origin main" in source:
            print(f"Removing redundant git pull at index {i}")
            is_redundant = True
            
        # Redundant Evaluation (covered in the loop)
        elif "Advanced Evaluation" in source and cell['cell_type'] == 'markdown':
             print(f"Removing redundant Markdown 'Advanced Evaluation' at index {i}")
             is_redundant = True
        elif "evaluate_ensemble.py" in source:
             print(f"Removing redundant evaluate_ensemble execution at index {i}")
             is_redundant = True
        elif "evaluate_uncertainty.py" in source:
             print(f"Removing redundant evaluate_uncertainty execution at index {i}")
             is_redundant = True
             
        # Javascript Alarm
        elif "google.com/sounds/v1/alarms/alarm_clock.ogg" in source:
            print(f"Removing alarm cell at index {i}")
            is_redundant = True

        if not is_redundant:
            new_cells.append(cell)
            
    # Logic for cells BEFORE the training loop
    else:
        # Keep everything else (setup, data loading, etc.)
        new_cells.append(cell)

nb['cells'] = new_cells
print(f"Refactoring complete. Cells reduced from {original_cell_count} to {len(new_cells)}.")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Notebook saved.")
