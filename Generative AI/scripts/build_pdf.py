"""Convert Generative_AI_Analysis_Report.md -> .pdf.

Uses markdown-pdf (PyMuPDF backend) which renders inline images and supports
basic tables. KaTeX math is left as raw $...$ which renders inline as text —
acceptable for the report's small number of equations.
"""
from pathlib import Path
from markdown_pdf import MarkdownPdf, Section

here = Path(__file__).resolve().parent.parent
md_path  = here / 'Generative_AI_Analysis_Report.md'
pdf_path = here / 'Generative_AI_Analysis_Report.pdf'

text = md_path.read_text(encoding='utf-8')

pdf = MarkdownPdf(toc_level=2, optimize=True)

css = """
@page { margin: 18mm 16mm; }
body { font-family: 'Segoe UI', 'Helvetica', sans-serif; font-size: 10.5pt;
       line-height: 1.45; color: #111; }
h1 { font-size: 20pt; border-bottom: 1pt solid #888; padding-bottom: 4pt; }
h2 { font-size: 14pt; margin-top: 18pt; border-bottom: 0.5pt solid #bbb;
     padding-bottom: 2pt; }
h3 { font-size: 11.5pt; margin-top: 12pt; }
table { border-collapse: collapse; margin: 8pt 0; font-size: 9.5pt; }
th, td { border: 0.5pt solid #888; padding: 4pt 8pt; text-align: left; }
th { background: #eee; }
code { font-family: 'Consolas', monospace; font-size: 9.5pt;
       background: #f4f4f4; padding: 1pt 3pt; border-radius: 2pt; }
img { max-width: 100%; }
"""

pdf.add_section(Section(text, root=str(here), paper_size='A4'), user_css=css)
pdf.meta['title']  = 'Generative AI Analysis Report'
pdf.meta['author'] = 'Project 5 - Generative AI Applications'
pdf.save(pdf_path)
print(f'wrote {pdf_path}  ({pdf_path.stat().st_size:,} bytes)')
