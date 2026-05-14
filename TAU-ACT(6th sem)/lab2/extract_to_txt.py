import pypdf

def extract(pdf_path, txt_path):
    try:
        reader = pypdf.PdfReader(pdf_path)
        with open(txt_path, 'w', encoding='utf-8') as f:
            for i, page in enumerate(reader.pages):
                f.write(f"\n\n--- PAGE {i+1} ---\n\n")
                t = page.extract_text()
                if t:
                    f.write(t)
    except Exception as e:
        print(f"Error {pdf_path}: {e}")

extract(r"c:\Users\Артем\Documents\ITMO-TAU-ACT\TAU-ACT(6th sem)\lab2\TAU_LR2 (2).pdf", r"c:\Users\Артем\tmp_req.txt")
extract(r"c:\Users\Артем\Documents\ITMO-TAU-ACT\TAU-ACT(6th sem)\lab2\examples\TAU2.pdf", r"c:\Users\Артем\tmp_ex.txt")
print("Done")
