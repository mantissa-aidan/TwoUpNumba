import subprocess
import os
import shutil

LATEX_FILENAME = "academic_paper_draft.tex"
PDF_FILENAME = "academic_paper_draft.pdf"
AUX_FILES_EXTENSIONS = [".aux", ".log", ".out", ".toc", ".lof", ".lot", ".fls", ".fdb_latexmk"]

def find_pdflatex():
    """Check if pdflatex command is available."""
    if shutil.which("pdflatex"):
        return "pdflatex"
    else:
        print("Error: pdflatex command not found. Please ensure a LaTeX distribution (like MiKTeX or TeX Live) is installed and in your system PATH.")
        return None

def compile_to_pdf(tex_filename):
    """Compiles the .tex file to .pdf using pdflatex."""
    pdflatex_cmd = find_pdflatex()
    if not pdflatex_cmd:
        return False

    if not os.path.exists(tex_filename):
        print(f"Error: LaTeX source file '{tex_filename}' not found.")
        return False

    # It's often necessary to run pdflatex twice for cross-references, table of contents, etc.
    # For simple documents, once might be enough, but twice is safer.
    # We'll use -interaction=nonstopmode to prevent it from stopping on minor errors
    # and -halt-on-error to stop on major errors.
    common_args = [pdflatex_cmd, "-interaction=nonstopmode", "-halt-on-error", tex_filename]
    
    print(f"Attempting to compile {tex_filename}...")
    
    success = True
    for i in range(2): # Run twice
        print(f"Compilation pass {i+1}...")
        try:
            # Capture output to check for common errors, though parsing LaTeX logs is complex.
            # For now, we mostly rely on the return code.
            process = subprocess.run(common_args, capture_output=True, text=True, check=False)
            
            if process.returncode != 0:
                print(f"--- pdflatex output (Pass {i+1}) ---")
                print(process.stdout)
                print(f"--- pdflatex errors (Pass {i+1}) ---")
                print(process.stderr)
                print(f"Error: pdflatex compilation failed on pass {i+1} with return code {process.returncode}.")
                print(f"Check '{os.path.splitext(tex_filename)[0]}.log' for detailed LaTeX errors.")
                success = False
                break # Stop if a pass fails
            else:
                print(f"Pass {i+1} completed.")

        except FileNotFoundError:
            print(f"Error: '{pdflatex_cmd}' command not found. Is LaTeX installed and in PATH?")
            success = False
            break
        except Exception as e:
            print(f"An unexpected error occurred during compilation: {e}")
            success = False
            break
            
    if success:
        print(f"Successfully compiled '{tex_filename}' to '{PDF_FILENAME}'.")
        # Clean up auxiliary files
        print("Cleaning up auxiliary files...")
        base_name = os.path.splitext(tex_filename)[0]
        for ext in AUX_FILES_EXTENSIONS:
            aux_file = base_name + ext
            if os.path.exists(aux_file):
                try:
                    os.remove(aux_file)
                except Exception as e:
                    print(f"Warning: Could not remove auxiliary file {aux_file}: {e}")
        return True
    else:
        print(f"PDF generation failed for '{tex_filename}'.")
        return False

if __name__ == "__main__":
    if compile_to_pdf(LATEX_FILENAME):
        print(f"'{PDF_FILENAME}' should now be available in your workspace.")
    else:
        print("Please check the console output for errors.") 