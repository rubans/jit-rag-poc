import fitz
import json

doc = fitz.open("data/documents/bp_annual_report_2023.pdf")

def check_snippet(page_no, needle):
    txt = doc[page_no - 1].get_text()
    clean_txt = " ".join(txt.split())
    found = needle.lower() in clean_txt.lower()
    print(f"Page {page_no}: {'OK' if found else 'NOT FOUND'}")
    if not found:
        print("  Looking for:", needle[:60])
        print("  Sample from page:", clean_txt[:120])

check_snippet(3, "87,800")
check_snippet(3, "$5.78/boe")
check_snippet(6, "Chair")
check_snippet(26, "Tier 1 process safety events")
check_snippet(38, "gas & low carbon energy, oil production & operations and customers & products")
check_snippet(115, "Bernard Looney resigned as CEO")
check_snippet(123, "Murray Auchincloss was appointed as CEO")
check_snippet(166, "210,130")
check_snippet(169, "280,294")
check_snippet(170, "32,039")
check_snippet(196, "14,080")
check_snippet(257, "Movements in estimated net proved reserves")
check_snippet(360, "restrictions on the Russian financial sector")
check_snippet(366, "Each ADS represents six ordinary shares")
check_snippet(389, "This document contains the Strategic report on the inside front cover and pages 1-80")

