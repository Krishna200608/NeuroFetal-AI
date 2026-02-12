import json
import os

nb_path = r"d:\Research Project\Research_Project\Code\notebooks\Training_Colab.ipynb"

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Check if cell already exists to avoid duplication if I run this twice
exists = False
for cell in nb['cells']:
    if "convert_to_tflite.py" in "".join(cell.get('source', [])):
        exists = True
        break

if not exists:
    new_cell = {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {
        "id": "run_tflite"
      },
      "outputs": [],
      "source": [
        "# Convert and Push TFLite Model\n",
        "!python Code/scripts/convert_to_tflite.py"
      ]
    }
    nb['cells'].append(new_cell)
    print("Added TFLite cell.")
else:
    print("TFLite cell already exists.")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)
