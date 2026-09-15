"""
Downloads the BP Annual Report and Form 20-F (2023) PDF into data/documents/.
"""

import urllib.request
import os
import sys
from pathlib import Path
import fitz  # PyMuPDF to verify pages and contents

URL = "https://www.annualreports.com/HostedData/AnnualReportArchive/b/LSE_BP_2023.pdf"
DEST_PATH = Path("data/documents/bp_annual_report_2023.pdf")

def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from: {url}")
    print(f"Destination: {dest.resolve()}")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req) as response, open(dest, 'wb') as out_file:
        total_size = response.getheader('Content-Length')
        if total_size:
            total_size = int(total_size)
            print(f"Total size: {total_size / (1024*1024):.2f} MB")
        
        bytes_downloaded = 0
        chunk_size = 1024 * 1024  # 1MB chunks
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            bytes_downloaded += len(chunk)
            if total_size:
                percent = (bytes_downloaded / total_size) * 100
                print(f"Downloaded: {bytes_downloaded / (1024*1024):.2f} MB ({percent:.1f}%)", end="\r")
            else:
                print(f"Downloaded: {bytes_downloaded / (1024*1024):.2f} MB", end="\r")
    
    print("\nDownload complete!")

def verify_pdf(path: Path) -> None:
    print(f"\nVerifying PDF: {path}...")
    doc = fitz.open(path)
    page_count = len(doc)
    print(f"Total page count: {page_count}")
    
    # Check sample pages for images and tables
    images_found = 0
    for page_idx in range(min(50, page_count)):
        page = doc[page_idx]
        image_list = page.get_images(full=True)
        images_found += len(image_list)
        
    print(f"Images found in first 50 pages: {images_found}")
    doc.close()

if __name__ == "__main__":
    download_file(URL, DEST_PATH)
    verify_pdf(DEST_PATH)
