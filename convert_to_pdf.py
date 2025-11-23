#!/usr/bin/env python3
"""
Convert FINAL_REPORT.md to PDF
Uses markdown2 and weasyprint or pdfkit
"""

import subprocess
import sys
import os

def check_and_install(package):
    """Check if package is installed, install if not"""
    try:
        __import__(package)
        return True
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
        return True

def convert_with_markdown_pdf():
    """Method 1: Using markdown-pdf (Node.js based)"""
    try:
        result = subprocess.run(['markdown-pdf', 'FINAL_REPORT.md', '-o', 'FINAL_REPORT.pdf'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return True
    except:
        pass
    return False

def convert_with_grip():
    """Method 2: Using grip to render and then print"""
    try:
        check_and_install('grip')
        import grip
        # This would require additional steps
        return False
    except:
        return False

def convert_with_python_libs():
    """Method 3: Using Python libraries"""
    try:
        # Try markdown2 + weasyprint
        check_and_install('markdown2')
        check_and_install('weasyprint')
        
        import markdown2
        from weasyprint import HTML, CSS
        from weasyprint.text.fonts import FontConfiguration
        
        print("Reading Markdown file...")
        with open('FINAL_REPORT.md', 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        print("Converting Markdown to HTML...")
        html_content = markdown2.markdown(md_content, extras=['tables', 'fenced-code-blocks', 'header-ids'])
        
        # Add CSS styling
        css_style = """
        <style>
            body {
                font-family: 'DejaVu Sans', Arial, sans-serif;
                line-height: 1.6;
                max-width: 800px;
                margin: 40px auto;
                padding: 20px;
                font-size: 11pt;
            }
            h1 { 
                font-size: 24pt; 
                margin-top: 20px;
                margin-bottom: 10px;
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 5px;
            }
            h2 { 
                font-size: 18pt; 
                margin-top: 16px;
                margin-bottom: 8px;
                color: #34495e;
            }
            h3 { 
                font-size: 14pt;
                margin-top: 12px;
                margin-bottom: 6px;
                color: #34495e;
            }
            h4 {
                font-size: 12pt;
                margin-top: 10px;
                margin-bottom: 5px;
                color: #555;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 15px 0;
                font-size: 10pt;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #3498db;
                color: white;
                font-weight: bold;
            }
            tr:nth-child(even) {
                background-color: #f2f2f2;
            }
            code {
                background-color: #f4f4f4;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
                font-size: 10pt;
            }
            pre {
                background-color: #f4f4f4;
                padding: 10px;
                border-radius: 5px;
                overflow-x: auto;
            }
            strong {
                color: #2c3e50;
            }
            em {
                color: #555;
            }
            ul, ol {
                margin: 10px 0;
                padding-left: 30px;
            }
            li {
                margin: 5px 0;
            }
            hr {
                border: none;
                border-top: 1px solid #ddd;
                margin: 20px 0;
            }
            @page {
                size: Letter;
                margin: 1in;
            }
        </style>
        """
        
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            {css_style}
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        print("Generating PDF...")
        font_config = FontConfiguration()
        HTML(string=full_html).write_pdf(
            'FINAL_REPORT.pdf',
            font_config=font_config
        )
        
        return True
        
    except Exception as e:
        print(f"Error with Python libraries: {e}")
        return False

def main():
    print("=" * 60)
    print("FINAL_REPORT.md → PDF Converter")
    print("=" * 60)
    print()
    
    if not os.path.exists('FINAL_REPORT.md'):
        print("ERROR: FINAL_REPORT.md not found!")
        return 1
    
    print("Attempting conversion using Python libraries...")
    if convert_with_python_libs():
        print()
        print("✓ SUCCESS! PDF created: FINAL_REPORT.pdf")
        print()
        file_size = os.path.getsize('FINAL_REPORT.pdf') / 1024
        print(f"  File size: {file_size:.1f} KB")
        print()
        return 0
    
    print()
    print("=" * 60)
    print("Alternative method: Use online converter")
    print("=" * 60)
    print()
    print("Option 1: Upload FINAL_REPORT.md to:")
    print("  • https://www.markdowntopdf.com/")
    print("  • https://md2pdf.netlify.app/")
    print("  • https://dillinger.io/ (export as PDF)")
    print()
    print("Option 2: Use VS Code extension:")
    print("  • Install 'Markdown PDF' extension")
    print("  • Right-click on FINAL_REPORT.md")
    print("  • Select 'Markdown PDF: Export (pdf)'")
    print()
    print("Option 3: Manual command (if you have pandoc):")
    print("  pandoc FINAL_REPORT.md -o FINAL_REPORT.pdf --pdf-engine=pdflatex")
    print()
    
    return 1

if __name__ == '__main__':
    sys.exit(main())
