#!/usr/bin/env python3
"""
Search for phrases in PDF files within a specific folder.
Requires: pip install pypdf
"""

import sys
import os
from pathlib import Path
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
    TimeoutError as FuturesTimeoutError,
)
from threading import Lock
import warnings
from datetime import datetime

# Suppress all pypdf warnings
warnings.filterwarnings("ignore")

# Try both libraries
try:
    from pypdf import PdfReader as PyPdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    from PyPDF2 import PdfReader as PyPDF2Reader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

if not PYPDF_AVAILABLE and not PYPDF2_AVAILABLE:
    print("Error: Neither pypdf nor PyPDF2 found.")
    print("Install with: pip install pypdf PyPDF2")
    sys.exit(1)

# Thread-safe printing and output management
print_lock = Lock()
TERMINAL_LINES_THRESHOLD = 50  # When output exceeds this, write to file

class OutputManager:
    """Manages output to both terminal and file."""
    def __init__(self, search_phrase):
        self.lines = []
        self.line_count = 0
        self.file_output = False
        self.output_file = None
        self.search_phrase = search_phrase
        
    def add_line(self, text):
        """Add a line to output."""
        self.lines.append(text)
        self.line_count += 1
        
        # Switch to file output if threshold exceeded
        if not self.file_output and self.line_count > TERMINAL_LINES_THRESHOLD:
            self._switch_to_file()
        
        if self.file_output:
            self.output_file.write(text + '\n')
            self.output_file.flush()
        else:
            print(text)
    
    def _switch_to_file(self):
        """Switch from terminal to file output."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_phrase = "".join(c if c.isalnum() else "_" for c in self.search_phrase[:20])
        filename = f"search_results_{safe_phrase}_{timestamp}.txt"
        
        self.output_file = open(filename, 'w', encoding='utf-8')
        self.file_output = True
        
        # Write existing lines to file
        for line in self.lines:
            self.output_file.write(line + '\n')
        
        # Notify user
        print(f"\n{'='*80}")
        print(f"📝 Output exceeded {TERMINAL_LINES_THRESHOLD} lines - writing to file:")
        print(f"   {filename}")
        print(f"{'='*80}\n")
        self.output_filename = filename
    
    def close(self):
        """Close the output file if open."""
        if self.output_file:
            self.output_file.close()
    
    def get_filename(self):
        """Get the output filename if file output was used."""
        return self.output_filename if self.file_output else None


def safe_print(output_mgr, *args, **kwargs):
    """Thread-safe print function that uses OutputManager."""
    text = ' '.join(str(arg) for arg in args)
    with print_lock:
        output_mgr.add_line(text)


def search_pdf(
    pdf_path,
    search_phrase,
    case_sensitive=False,
    all_occurrences=False,
    debug=False,
    compare_libs=False,
):
    """
    Search for a phrase in a PDF file.

    Args:
        pdf_path: Path to the PDF file
        search_phrase: Phrase to search for
        case_sensitive: Whether search should be case-sensitive
        all_occurrences: If True, find all occurrences per page. If False, only first per page.
        debug: If True, print debug information
        compare_libs: If True, compare pypdf vs PyPDF2 extraction

    Returns:
        Tuple of (pdf_path, results_list, error_message)
    """
    results = []
    error_msg = None

    try:
        with open(pdf_path, "rb") as file:
            # Choose reader - prefer PyPDF2 for better text extraction
            if compare_libs and PYPDF_AVAILABLE and PYPDF2_AVAILABLE:
                reader_pypdf2 = PyPDF2Reader(file)
                file.seek(0)
                reader_pypdf = PyPdfReader(file)
                file.seek(0)
                reader = reader_pypdf2
            elif PYPDF2_AVAILABLE:
                reader = PyPDF2Reader(file)
            else:
                reader = PyPdfReader(file)

            total_pages = len(reader.pages)
            search_term = search_phrase if case_sensitive else search_phrase.lower()

            for page_num in range(total_pages):
                try:
                    # Suppress stderr only during text extraction
                    old_stderr = sys.stderr
                    sys.stderr = open(os.devnull, "w")
                    try:
                        if compare_libs and PYPDF_AVAILABLE and PYPDF2_AVAILABLE:
                            page_pypdf2 = reader_pypdf2.pages[page_num]
                            page_pypdf = reader_pypdf.pages[page_num]
                            text_pypdf2 = page_pypdf2.extract_text()
                            text_pypdf = (
                                page_pypdf.extract_text(extraction_mode="layout")
                                if hasattr(page_pypdf, "extract_text")
                                else page_pypdf.extract_text()
                            )
                            text = text_pypdf2
                        else:
                            page = reader.pages[page_num]
                            if PYPDF2_AVAILABLE:
                                text = page.extract_text()
                            else:
                                if hasattr(page, "extract_text"):
                                    text = page.extract_text(extraction_mode="layout")
                                else:
                                    text = page.extract_text()
                    finally:
                        sys.stderr.close()
                        sys.stderr = old_stderr

                    if not text:
                        continue

                    # Normalize whitespace in text
                    text_normalized = " ".join(text.split())

                    search_text = (
                        text_normalized if case_sensitive else text_normalized.lower()
                    )
                    search_term_normalized = " ".join(search_term.split())

                    # Quick check if term exists before processing
                    if search_term_normalized not in search_text:
                        continue

                    if all_occurrences:
                        pos = 0
                        while True:
                            index = search_text.find(search_term_normalized, pos)
                            if index == -1:
                                break

                            context_start = max(0, index - 60)
                            context_end = min(
                                len(text_normalized),
                                index + len(search_term_normalized) + 60,
                            )
                            context = text_normalized[context_start:context_end]

                            try:
                                context = context.encode("utf-8", "ignore").decode("utf-8")
                            except:
                                context = "".join(c if ord(c) < 128 else "?" for c in context)

                            if len(context) > 150:
                                context = context[:150] + "..."

                            results.append((page_num + 1, context))
                            pos = index + 1
                    else:
                        index = search_text.find(search_term_normalized)
                        if index != -1:
                            context_start = max(0, index - 60)
                            context_end = min(
                                len(text_normalized),
                                index + len(search_term_normalized) + 60,
                            )
                            context = text_normalized[context_start:context_end]

                            try:
                                context = context.encode("utf-8", "ignore").decode("utf-8")
                            except:
                                context = "".join(c if ord(c) < 128 else "?" for c in context)

                            if len(context) > 150:
                                context = context[:150] + "..."

                            results.append((page_num + 1, context))

                except Exception:
                    continue

    except Exception as e:
        error_msg = f"File error: {str(e)[:80]}"

    return (pdf_path, results, error_msg)


def search_folder(
    folder_path,
    search_phrase,
    case_sensitive=False,
    max_workers=4,
    all_occurrences=False,
    debug=False,
    compare_libs=False,
):
    """Search for a phrase in all PDF files in a folder using multiple threads."""
    folder = Path(folder_path).expanduser()

    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist")
        return

    if not folder.is_dir():
        print(f"Error: '{folder_path}' is not a directory")
        return

    pdf_files = sorted(list(folder.glob("*.pdf")))

    if not pdf_files:
        print(f"No PDF files found in '{folder_path}'")
        return

    # Create output manager
    output_mgr = OutputManager(search_phrase)

    output_mgr.add_line(f"\n🔍 Searching for '{search_phrase}' in {len(pdf_files)} PDF files...")
    output_mgr.add_line(f"⚙️  Using {max_workers} parallel threads")
    output_mgr.add_line(f"📚 Library: {'PyPDF2' if PYPDF2_AVAILABLE else 'pypdf'}")
    if compare_libs and PYPDF_AVAILABLE and PYPDF2_AVAILABLE:
        output_mgr.add_line(f"🔬 Comparing PyPDF2 vs pypdf extraction")
    output_mgr.add_line(f"Case sensitive: {case_sensitive}")
    output_mgr.add_line(f"Mode: {'All occurrences per page' if all_occurrences else 'First occurrence per page only'}")
    if debug:
        output_mgr.add_line(f"🐛 Debug mode: ON")
    output_mgr.add_line("=" * 80)
    output_mgr.add_line("")

    total_matches = 0
    processed = 0
    errors = []
    all_results = {}
    file_index = {}
    file_counter = [0]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pdf = {}
        for pdf_file in pdf_files:
            future = executor.submit(
                search_pdf,
                pdf_file,
                search_phrase,
                case_sensitive,
                all_occurrences,
                debug,
                compare_libs,
            )
            future_to_pdf[future] = pdf_file

        for future in as_completed(future_to_pdf, timeout=None):
            pdf_file = future_to_pdf[future]

            try:
                pdf_path, results, error_msg = future.result(timeout=30)

                processed += 1
                safe_print(output_mgr, f"[{processed}/{len(pdf_files)}] Processing: {pdf_file.name}")

                if error_msg:
                    errors.append((pdf_file.name, error_msg))
                elif results:
                    file_counter[0] += 1
                    file_label = file_counter[0]
                    file_index[file_label] = pdf_file

                    safe_print(output_mgr, f"\n{'=' * 80}")
                    safe_print(output_mgr, f"[{file_label}] {pdf_file.name}")
                    safe_print(output_mgr, f"{'=' * 80}")
                    safe_print(output_mgr, f"Found {len(results)} match(es):\n")
                    for page_num, context in results:
                        safe_print(output_mgr, f"  Page {page_num}:")
                        safe_print(output_mgr, f"    {context}\n")

                    all_results[pdf_file] = results
                    total_matches += len(results)

            except FuturesTimeoutError:
                processed += 1
                safe_print(output_mgr, f"[{processed}/{len(pdf_files)}] ⏱️  {pdf_file.name}: Timeout after 30s (skipping)")
                errors.append((pdf_file.name, "Timeout after 30s"))
                future.cancel()
            except Exception as e:
                processed += 1
                safe_print(output_mgr, f"[{processed}/{len(pdf_files)}] ❌ {pdf_file.name}: {str(e)[:80]}")
                errors.append((pdf_file.name, str(e)[:80]))

    output_mgr.add_line("\n" + "=" * 80)
    output_mgr.add_line(f"✅ Search complete!")
    output_mgr.add_line(f"   📊 {total_matches} total match(es) in {len(all_results)} file(s)")
    output_mgr.add_line(f"   📁 {processed} files processed")
    if errors:
        output_mgr.add_line(f"   ⚠️  {len(errors)} file(s) had errors")
    output_mgr.add_line("=" * 80)

    # Close output file if it was created
    output_mgr.close()
    output_filename = output_mgr.get_filename()

    # Offer to open the file if it was created
    if output_filename:
        print(f"\n💡 Results saved to: {output_filename}")
        print(f"Commands to view:")
        print(f"  cat {output_filename}")
        print(f"  less {output_filename}")
        print(f"  open {output_filename}  # macOS")
        print(f"  nano {output_filename}")
        
        try:
            choice = input("\nOpen file now? (y/n): ").strip().lower()
            if choice == 'y':
                import subprocess
                if sys.platform == 'darwin':  # macOS
                    subprocess.run(['open', output_filename])
                elif sys.platform == 'linux':
                    subprocess.run(['xdg-open', output_filename])
                elif sys.platform == 'win32':
                    subprocess.run(['notepad', output_filename])
        except KeyboardInterrupt:
            print("\n")

    # Interactive file opening for matched PDFs
    if file_index:
        print(f"\n💡 To open a PDF with matches, enter its number [1-{len(file_index)}] (or 'q' to quit):")
        while True:
            try:
                choice = input("Enter file number: ").strip()
                if choice.lower() == 'q':
                    break

                file_num = int(choice)
                if file_num in file_index:
                    pdf_file = file_index[file_num]
                    print(f"Opening {pdf_file.name}...")

                    import subprocess
                    subprocess.run(["open", "-a", "Preview", str(pdf_file)])
                    print(f"✓ Opened!\n")
                else:
                    print(f"Invalid file number. Choose between 1 and {len(file_index)}")
            except ValueError:
                print("Please enter a valid number or 'q' to quit")
            except KeyboardInterrupt:
                print("\n")
                break


def generate_html_report(
    folder_path,
    search_phrase,
    case_sensitive=False,
    max_workers=4,
    all_occurrences=False,
):
    """Generate an HTML report of search results using multi-threading."""
    folder = Path(folder_path).expanduser()
    pdf_files = sorted(list(folder.glob("*.pdf")))

    print(f"\n🔍 Searching {len(pdf_files)} PDFs with {max_workers} threads...")
    print(f"Mode: {'All occurrences per page' if all_occurrences else 'First occurrence per page only'}")

    all_results = {}
    total_matches = 0
    errors = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pdf = {}
        for pdf_file in pdf_files:
            future = executor.submit(
                search_pdf, pdf_file, search_phrase, case_sensitive, all_occurrences
            )
            future_to_pdf[future] = pdf_file

        for future in as_completed(future_to_pdf, timeout=None):
            pdf_file = future_to_pdf[future]
            try:
                pdf_path, results, error_msg = future.result(timeout=30)
                if error_msg:
                    errors.append((pdf_file.name, error_msg))
                elif results:
                    all_results[pdf_file] = results
                    total_matches += len(results)
            except FuturesTimeoutError:
                errors.append((pdf_file.name, "Timeout after 30s"))
            except Exception as e:
                errors.append((pdf_file.name, str(e)))

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>PDF Search Results</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
               max-width: 1200px; margin: 40px auto; padding: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        .search-info {{ background: #e3f2fd; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        .pdf-result {{ background: white; margin: 20px 0; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .pdf-title {{ color: #1976d2; font-size: 1.2em; font-weight: bold; margin-bottom: 10px; }}
        .match {{ margin: 10px 0; padding: 10px; background: #f9f9f9; border-left: 3px solid #4caf50; }}
        .page-num {{ color: #666; font-weight: bold; }}
        .context {{ color: #333; font-family: monospace; font-size: 0.9em; }}
        .highlight {{ background-color: #ffeb3b; padding: 2px 4px; }}
        .stats {{ color: #666; margin-top: 20px; padding: 15px; background: white; border-radius: 8px; }}
        .open-link {{ color: #1976d2; text-decoration: none; font-size: 0.9em; }}
        .open-link:hover {{ text-decoration: underline; }}
        .errors {{ background: #ffebee; padding: 15px; border-radius: 8px; margin: 20px 0; }}
        .error-item {{ color: #c62828; margin: 5px 0; }}
    </style>
</head>
<body>
    <h1>🔍 PDF Search Results</h1>
    <div class="search-info">
        <strong>Search phrase:</strong> "{search_phrase}"<br>
        <strong>Folder:</strong> {folder.absolute()}<br>
        <strong>Case sensitive:</strong> {case_sensitive}<br>
        <strong>Search mode:</strong> {"All occurrences per page" if all_occurrences else "First occurrence per page only"}<br>
        <strong>Files searched:</strong> {len(pdf_files)}<br>
        <strong>Threads used:</strong> {max_workers}
    </div>
"""

    if errors:
        html_content += """
    <div class="errors">
        <strong>⚠️ Errors encountered:</strong><br>
"""
        for filename, error in errors:
            html_content += f'        <div class="error-item">• {filename}: {error}</div>\n'
        html_content += "    </div>\n"

    for pdf_file in sorted(all_results.keys(), key=lambda x: x.name):
        results = all_results[pdf_file]
        html_content += f"""
    <div class="pdf-result">
        <div class="pdf-title">📄 {pdf_file.name}</div>
        <a class="open-link" href="file://{pdf_file.absolute()}">Open PDF</a>
        <div style="margin-top: 10px;"><strong>{len(results)} match(es) found:</strong></div>
"""
        for page_num, context in results:
            if case_sensitive:
                highlighted = context.replace(
                    search_phrase, f'<span class="highlight">{search_phrase}</span>'
                )
            else:
                import re
                highlighted = re.sub(
                    f"({re.escape(search_phrase)})",
                    r'<span class="highlight">\1</span>',
                    context,
                    flags=re.IGNORECASE,
                )

            html_content += f"""
        <div class="match">
            <span class="page-num">Page {page_num}:</span>
            <div class="context">...{highlighted}...</div>
        </div>
"""

        html_content += "    </div>\n"

    html_content += f"""
    <div class="stats">
        <strong>Search complete:</strong> {total_matches} total match(es) found in {len(all_results)} file(s)<br>
        <strong>Processed:</strong> {len(pdf_files)} files
    </div>
</body>
</html>
"""

    report_path = Path("pdf_search_results.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n✅ HTML report generated: {report_path.absolute()}")

    import webbrowser
    webbrowser.open(f"file://{report_path.absolute()}")


def main():
    """Main function to handle command-line arguments."""

    default_folder = "./Algorithms"

    if len(sys.argv) < 2:
        print("Usage: python search_pdf.py <search_phrase> [folder_path] [options]")
        print(f"\nOptions:")
        print(f"  --case-sensitive    Search with case sensitivity")
        print(f"  --html             Generate HTML report")
        print(f"  --threads N        Use N parallel threads (1-16, default: 4)")
        print(f"  --all              Find ALL occurrences per page (default: first only)")
        print(f"  --debug            Show debug information for text extraction")
        print(f"  --compare          Compare pypdf vs PyPDF2 extraction accuracy")
        print(f"\nNote: If output exceeds 50 lines, results auto-save to a text file")
        print(f"\nExamples:")
        print(f"  python search_pdf.py 'dynamic programming' '{default_folder}'")
        print(f"  python search_pdf.py 'Algorithm' --case-sensitive")
        print(f"  python search_pdf.py 'parallel' --html --threads 8")
        print(f"  python search_pdf.py 'majority element' --all")
        sys.exit(1)

    search_phrase = sys.argv[1]

    if len(sys.argv) >= 3 and not sys.argv[2].startswith("--"):
        folder_path = sys.argv[2]
    else:
        folder_path = default_folder

    case_sensitive = "--case-sensitive" in sys.argv
    html_output = "--html" in sys.argv
    all_occurrences = "--all" in sys.argv
    debug = "--debug" in sys.argv
    compare_libs = "--compare" in sys.argv

    max_workers = 4
    if "--threads" in sys.argv:
        try:
            thread_idx = sys.argv.index("--threads")
            if thread_idx + 1 < len(sys.argv):
                max_workers = int(sys.argv[thread_idx + 1])
                max_workers = max(1, min(max_workers, 16))
        except (ValueError, IndexError):
            print("Warning: Invalid thread count, using default (4)")

    if html_output:
        generate_html_report(
            folder_path, search_phrase, case_sensitive, max_workers, all_occurrences
        )
    else:
        search_folder(
            folder_path,
            search_phrase,
            case_sensitive,
            max_workers,
            all_occurrences,
            debug,
            compare_libs,
        )


if __name__ == "__main__":
    main()