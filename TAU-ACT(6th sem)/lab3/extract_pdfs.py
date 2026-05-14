import os
import sys
import subprocess

def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

install_and_import('pypdf')
import pypdf

def extract_pdf_to_text(pdf_path, output_txt_path):
    print(f"Extracting {pdf_path}...")
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text.append(t)
        
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write("\n\n--- PAGE BREAK ---\n\n".join(text))
        print(f"Saved to {output_txt_path}")
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")

if __name__ == "__main__":
    lab3_dir = r"c:\Users\Артем\Documents\ITMO-TAU-ACT\TAU-ACT(6th sem)\lab3\examples"
    out_dir = r"c:\Users\Артем\Documents\ITMO-TAU-ACT\TAU-ACT(6th sem)\lab3\pdf_texts"
    os.makedirs(out_dir, exist_ok=True)
    
    for fname in os.listdir(lab3_dir):
        if fname.lower().endswith('.pdf'):
            pdf_path = os.path.join(lab3_dir, fname)
            out_path = os.path.join(out_dir, fname + ".txt")
            extract_pdf_to_text(pdf_path, out_path)
