"""
Export IPUESA paper to styled HTML (can be printed to PDF from browser)
"""

import markdown
from pathlib import Path
import re
import base64

def image_to_base64(image_path: Path) -> str:
    """Convert image to base64 for embedding in HTML"""
    if image_path.exists():
        with open(image_path, 'rb') as f:
            data = base64.b64encode(f.read()).decode('utf-8')
            suffix = image_path.suffix.lower()
            mime = 'image/png' if suffix == '.png' else 'image/jpeg'
            return f'data:{mime};base64,{data}'
    return ''

def markdown_to_html(md_content: str, figures_dir: Path) -> str:
    """Convert markdown to HTML with embedded images and styling"""

    # Find all image references and embed them as base64
    def replace_image(match):
        alt_text = match.group(1)
        img_name = match.group(2)
        img_path = figures_dir / img_name
        base64_data = image_to_base64(img_path)
        if base64_data:
            return f'<img src="{base64_data}" alt="{alt_text}" />'
        return match.group(0)

    md_content = re.sub(
        r'!\[(.*?)\]\(figures/(.*?)\)',
        replace_image,
        md_content
    )

    # Convert markdown to HTML
    html_body = markdown.markdown(
        md_content,
        extensions=['tables', 'fenced_code', 'codehilite', 'md_in_html']
    )

    # Full HTML with print-optimized CSS
    html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="utf-8">
    <title>IPUESA: Atractores de Identidad Funcional</title>
    <style>
        /* Print-optimized styles */
        @media print {{
            body {{
                font-size: 10pt;
            }}
            h1 {{
                font-size: 16pt;
            }}
            h2 {{
                font-size: 13pt;
                page-break-after: avoid;
            }}
            h3 {{
                font-size: 11pt;
                page-break-after: avoid;
            }}
            img {{
                max-height: 400px;
                page-break-inside: avoid;
            }}
            table {{
                page-break-inside: avoid;
            }}
            pre {{
                page-break-inside: avoid;
            }}
        }}

        @page {{
            size: A4;
            margin: 2cm 2cm;
        }}

        * {{
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Georgia', 'Times New Roman', 'Palatino', serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #1a1a1a;
            max-width: 800px;
            margin: 0 auto;
            padding: 2em;
            background: white;
        }}

        h1 {{
            font-size: 20pt;
            font-weight: bold;
            color: #1a5276;
            margin-top: 0;
            margin-bottom: 0.8em;
            text-align: center;
            border-bottom: 3px solid #1a5276;
            padding-bottom: 0.5em;
        }}

        h2 {{
            font-size: 14pt;
            font-weight: bold;
            color: #1a5276;
            margin-top: 1.8em;
            margin-bottom: 0.6em;
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 0.3em;
        }}

        h3 {{
            font-size: 12pt;
            font-weight: bold;
            color: #2c3e50;
            margin-top: 1.2em;
            margin-bottom: 0.4em;
        }}

        p {{
            margin-bottom: 0.9em;
            text-align: justify;
            hyphens: auto;
        }}

        strong {{
            color: #1a1a1a;
            font-weight: 600;
        }}

        blockquote {{
            border-left: 4px solid #1a5276;
            padding: 0.8em 1.2em;
            margin: 1em 0;
            margin-left: 0;
            background: #f8f9fa;
            font-style: italic;
            color: #2c3e50;
        }}

        blockquote p {{
            margin-bottom: 0;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1.2em 0;
            font-size: 10pt;
        }}

        th {{
            background-color: #1a5276;
            color: white;
            padding: 10px 12px;
            text-align: left;
            font-weight: 600;
        }}

        td {{
            padding: 8px 12px;
            border-bottom: 1px solid #ddd;
            vertical-align: top;
        }}

        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}

        code {{
            font-family: 'Consolas', 'Monaco', 'Menlo', monospace;
            font-size: 9pt;
            background: #ecf0f1;
            padding: 2px 6px;
            border-radius: 3px;
            color: #c0392b;
        }}

        pre {{
            background: #2c3e50;
            color: #ecf0f1;
            padding: 1em 1.2em;
            border-radius: 5px;
            overflow-x: auto;
            font-size: 9pt;
            line-height: 1.4;
            margin: 1em 0;
        }}

        pre code {{
            background: none;
            padding: 0;
            color: #ecf0f1;
            border-radius: 0;
        }}

        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 1.5em auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}

        /* Figure caption styling */
        p > em:only-child {{
            display: block;
            text-align: center;
            font-size: 9pt;
            color: #666;
            margin-top: -1em;
            margin-bottom: 1.5em;
        }}

        em {{
            font-style: italic;
            color: #555;
        }}

        hr {{
            border: none;
            border-top: 2px solid #ecf0f1;
            margin: 2.5em 0;
        }}

        ul, ol {{
            margin-bottom: 1em;
            padding-left: 1.8em;
        }}

        li {{
            margin-bottom: 0.4em;
        }}

        /* Print button - hidden when printing */
        .print-button {{
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 12px 24px;
            background: #1a5276;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }}

        .print-button:hover {{
            background: #154360;
        }}

        @media print {{
            .print-button {{
                display: none;
            }}
        }}

        /* Abstract box */
        p:first-of-type {{
            background: #f8f9fa;
            padding: 1em 1.2em;
            border-left: 4px solid #1a5276;
            margin-bottom: 1.5em;
            font-size: 10pt;
        }}
    </style>
</head>
<body>
    <button class="print-button" onclick="window.print()">Imprimir / Guardar PDF</button>
    {html_body}
    <script>
        // Auto-focus for print
        console.log('Paper loaded. Click the button or press Ctrl+P to print/save as PDF.');
    </script>
</body>
</html>
"""

    return html


def export_to_html(md_path: str, html_path: str):
    """Export markdown file to HTML"""

    md_path = Path(md_path)
    html_path = Path(html_path)
    figures_dir = md_path.parent / 'figures'

    print(f"Reading: {md_path}")
    md_content = md_path.read_text(encoding='utf-8')

    print(f"Figures directory: {figures_dir}")
    print(f"Figures found: {list(figures_dir.glob('*.png'))}")

    print("Converting to HTML with embedded images...")
    html_content = markdown_to_html(md_content, figures_dir)

    html_path.write_text(html_content, encoding='utf-8')
    print(f"\nHTML saved: {html_path}")
    print("\nPara generar PDF:")
    print("  1. Abre el archivo HTML en tu navegador")
    print("  2. Click en 'Imprimir / Guardar PDF' o presiona Ctrl+P")
    print("  3. Selecciona 'Guardar como PDF' como destino")

    return html_path


def main():
    base_dir = Path(__file__).parent.parent

    md_path = base_dir / 'docs' / 'papers' / 'ipuesa-identidad-funcional-paper.md'
    html_path = base_dir / 'docs' / 'papers' / 'ipuesa-identidad-funcional-paper.html'

    export_to_html(md_path, html_path)
    print("\nDone!")


if __name__ == '__main__':
    main()
