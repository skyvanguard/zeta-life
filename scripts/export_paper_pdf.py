"""
Export IPUESA paper to PDF using markdown + weasyprint
"""

import markdown
from weasyprint import HTML, CSS
from pathlib import Path
import re

def markdown_to_html(md_content: str, figures_dir: str) -> str:
    """Convert markdown to HTML with styling"""

    # Fix image paths to be absolute
    md_content = re.sub(
        r'!\[(.*?)\]\(figures/(.*?)\)',
        rf'![\1]({figures_dir}/\2)',
        md_content
    )

    # Convert markdown to HTML
    html_body = markdown.markdown(
        md_content,
        extensions=['tables', 'fenced_code', 'codehilite']
    )

    # Full HTML with CSS styling
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            @page {{
                size: A4;
                margin: 2cm 2.5cm;
                @top-center {{
                    content: "IPUESA: Atractores de Identidad Funcional";
                    font-size: 9pt;
                    color: #666;
                }}
                @bottom-center {{
                    content: counter(page);
                    font-size: 9pt;
                }}
            }}

            body {{
                font-family: 'Georgia', 'Times New Roman', serif;
                font-size: 11pt;
                line-height: 1.5;
                color: #1a1a1a;
                max-width: 100%;
            }}

            h1 {{
                font-size: 18pt;
                font-weight: bold;
                color: #2E86AB;
                margin-top: 0;
                margin-bottom: 1em;
                text-align: center;
                border-bottom: 2px solid #2E86AB;
                padding-bottom: 0.5em;
            }}

            h2 {{
                font-size: 14pt;
                font-weight: bold;
                color: #2E86AB;
                margin-top: 1.5em;
                margin-bottom: 0.5em;
                border-bottom: 1px solid #ddd;
                padding-bottom: 0.3em;
            }}

            h3 {{
                font-size: 12pt;
                font-weight: bold;
                color: #333;
                margin-top: 1em;
                margin-bottom: 0.5em;
            }}

            p {{
                margin-bottom: 0.8em;
                text-align: justify;
            }}

            strong {{
                color: #1a1a1a;
            }}

            blockquote {{
                border-left: 3px solid #2E86AB;
                padding-left: 1em;
                margin-left: 0;
                color: #444;
                font-style: italic;
                background: #f8f9fa;
                padding: 0.5em 1em;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 1em 0;
                font-size: 10pt;
            }}

            th {{
                background-color: #2E86AB;
                color: white;
                padding: 8px 12px;
                text-align: left;
                font-weight: bold;
            }}

            td {{
                padding: 6px 12px;
                border-bottom: 1px solid #ddd;
            }}

            tr:nth-child(even) {{
                background-color: #f8f9fa;
            }}

            code {{
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 9pt;
                background: #f4f4f4;
                padding: 2px 5px;
                border-radius: 3px;
            }}

            pre {{
                background: #2d2d2d;
                color: #f8f8f2;
                padding: 1em;
                border-radius: 5px;
                overflow-x: auto;
                font-size: 9pt;
                line-height: 1.4;
            }}

            pre code {{
                background: none;
                padding: 0;
                color: #f8f8f2;
            }}

            img {{
                max-width: 100%;
                height: auto;
                display: block;
                margin: 1em auto;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}

            em {{
                font-style: italic;
                color: #555;
            }}

            hr {{
                border: none;
                border-top: 1px solid #ddd;
                margin: 2em 0;
            }}

            ul, ol {{
                margin-bottom: 1em;
                padding-left: 2em;
            }}

            li {{
                margin-bottom: 0.3em;
            }}

            /* Figure captions */
            img + em {{
                display: block;
                text-align: center;
                font-size: 9pt;
                color: #666;
                margin-top: -0.5em;
                margin-bottom: 1em;
            }}

            /* Abstract styling */
            p:first-of-type {{
                font-size: 10pt;
            }}
        </style>
    </head>
    <body>
        {html_body}
    </body>
    </html>
    """

    return html


def export_to_pdf(md_path: str, pdf_path: str):
    """Export markdown file to PDF"""

    md_path = Path(md_path)
    pdf_path = Path(pdf_path)
    figures_dir = md_path.parent / 'figures'

    print(f"Reading: {md_path}")
    md_content = md_path.read_text(encoding='utf-8')

    print("Converting to HTML...")
    html_content = markdown_to_html(md_content, str(figures_dir.absolute()))

    # Save HTML for debugging
    html_path = pdf_path.with_suffix('.html')
    html_path.write_text(html_content, encoding='utf-8')
    print(f"HTML saved: {html_path}")

    print("Generating PDF...")
    HTML(string=html_content, base_url=str(md_path.parent.absolute())).write_pdf(pdf_path)

    print(f"PDF saved: {pdf_path}")
    return pdf_path


def main():
    base_dir = Path(__file__).parent.parent

    md_path = base_dir / 'docs' / 'papers' / 'ipuesa-identidad-funcional-paper.md'
    pdf_path = base_dir / 'docs' / 'papers' / 'ipuesa-identidad-funcional-paper.pdf'

    export_to_pdf(md_path, pdf_path)
    print("\nDone!")


if __name__ == '__main__':
    main()
