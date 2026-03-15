import os
import nbformat
from nbconvert import PythonExporter

def convert_notebooks(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    exporter = PythonExporter()

    for filename in os.listdir(input_folder):
        if filename.endswith(".ipynb"):
            notebook_path = os.path.join(input_folder, filename)

            # Read notebook
            with open(notebook_path, "r", encoding="utf-8") as f:
                notebook = nbformat.read(f, as_version=4)

            # Convert to python script
            script, _ = exporter.from_notebook_node(notebook)

            # Output filename
            output_filename = filename.replace(".ipynb", ".py")
            output_path = os.path.join(output_folder, output_filename)

            # Save python file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(script)

            print(f"Converted: {filename} → {output_filename}")

    print("All notebooks converted successfully.")


if __name__ == "__main__":
    input_folder = r"all_rag_techniques"
    output_folder = r"all_rag_techniques_scripts"

    convert_notebooks(input_folder, output_folder)