#!/usr/bin/env python3
"""
Phase 1: Scrape 2026 AIME I problems and answers from AoPS wiki.
"""
import json
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

URL = "https://artofproblemsolving.com/wiki/index.php/2026_AIME_I"
OUTPUT = Path(__file__).parent.parent / "data" / "aime_2026_i.json"


def fetch_page(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; AIME-scraper/1.0)"}
    for attempt in range(2):
        try:
            resp = requests.get(url, headers=headers, timeout=120)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            if attempt == 0:
                print(f"  Attempt 1 failed: {e}. Retrying...")
            else:
                raise RuntimeError(
                    f"Failed to fetch {url}: {e}\n"
                    "If AoPS is unavailable, manually populate data/aime_2026_i.json."
                ) from e


def extract_math_text(tag) -> str:
    """Convert a tag's contents to text, preserving LaTeX from <math> tags."""
    parts = []
    for child in tag.children:
        if hasattr(child, "name"):
            if child.name == "math":
                latex = child.get_text()
                parts.append(f"${latex}$")
            elif child.name == "img" and child.get("alt"):
                parts.append(child["alt"])
            else:
                parts.append(extract_math_text(child))
        else:
            parts.append(str(child))
    return "".join(parts)


def parse_problems(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    content = soup.find("div", class_="mw-parser-output")
    if not content:
        raise RuntimeError(
            "Could not find .mw-parser-output div. AoPS page structure may have changed.\n"
            "Please manually populate data/aime_2026_i.json."
        )

    problems = []
    current_problem_num = None
    current_paragraphs = []
    in_solution = False

    for tag in content.children:
        if not hasattr(tag, "name"):
            continue

        if tag.name in ("h2", "h3"):
            # Save previous problem if we were collecting one
            if current_problem_num is not None and current_paragraphs:
                text = " ".join(current_paragraphs).strip()
                problems.append({"num": current_problem_num, "problem": text})
                current_paragraphs = []

            heading_text = tag.get_text(strip=True)

            # Check for Problem N heading
            m = re.match(r"Problem\s+(\d+)", heading_text, re.IGNORECASE)
            if m:
                current_problem_num = int(m.group(1))
                in_solution = False
            elif re.match(r"Solution", heading_text, re.IGNORECASE):
                in_solution = True
                if current_problem_num is not None and current_paragraphs:
                    text = " ".join(current_paragraphs).strip()
                    problems.append({"num": current_problem_num, "problem": text})
                    current_paragraphs = []
                    current_problem_num = None
            else:
                current_problem_num = None

        elif tag.name == "p" and current_problem_num is not None and not in_solution:
            text = extract_math_text(tag).strip()
            if text:
                current_paragraphs.append(text)

    # Catch last problem if page ends without Solution heading
    if current_problem_num is not None and current_paragraphs:
        text = " ".join(current_paragraphs).strip()
        problems.append({"num": current_problem_num, "problem": text})

    return problems


def parse_answers(html: str) -> dict[int, int]:
    """
    Try to extract answers from the answer key section of the page.
    AoPS often has a section like '== Answer Key ==' or similar,
    or answers embedded in solution pages. We look for patterns like
    'Answer: 042' or AIME answer boxes.
    """
    soup = BeautifulSoup(html, "html.parser")
    content = soup.find("div", class_="mw-parser-output")
    answers = {}

    if not content:
        return answers

    in_answer_section = False
    for tag in content.children:
        if not hasattr(tag, "name"):
            continue

        if tag.name in ("h2", "h3"):
            heading = tag.get_text(strip=True).lower()
            if "answer" in heading:
                in_answer_section = True
            elif in_answer_section:
                in_answer_section = False

        if in_answer_section and tag.name in ("p", "li", "td"):
            text = tag.get_text(strip=True)
            # Look for patterns like "1. 042" or "Problem 1: 042"
            matches = re.findall(r"(?:Problem\s*)?(\d+)[.:\s]+(\d{1,3})\b", text)
            for prob_num, ans in matches:
                n = int(prob_num)
                a = int(ans)
                if 1 <= n <= 15 and 0 <= a <= 999:
                    answers[n] = a

    # Also look for answer key tables
    for table in content.find_all("table"):
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            cell_texts = [c.get_text(strip=True) for c in cells]
            # Look for rows with (number, answer) pairs
            if len(cell_texts) >= 2:
                for i in range(len(cell_texts) - 1):
                    m1 = re.match(r"^(\d+)$", cell_texts[i])
                    m2 = re.match(r"^(\d{1,3})$", cell_texts[i + 1])
                    if m1 and m2:
                        n = int(m1.group(1))
                        a = int(m2.group(1))
                        if 1 <= n <= 15 and 0 <= a <= 999:
                            answers[n] = a

    return answers


def main():
    # If data already exists and has 15 problems, skip scraping
    if OUTPUT.exists():
        try:
            existing = json.loads(OUTPUT.read_text())
            if len(existing.get("problems", [])) == 15:
                print(f"Data already present at {OUTPUT} (15 problems). Skipping scrape.")
                for p in existing["problems"]:
                    ans_str = str(p["answer"]) if p["answer"] is not None else "MISSING"
                    print(f"  Problem {p['id']:2d}: answer={ans_str}")
                return
        except (json.JSONDecodeError, KeyError):
            pass

    print(f"Fetching: {URL}")
    html = fetch_page(URL)
    print("Parsing problems...")
    problems = parse_problems(html)

    if not problems:
        print("ERROR: No problems found. Page structure may have changed.")
        print("Please manually populate data/aime_2026_i.json.")
        sys.exit(1)

    print(f"  Found {len(problems)} problems")

    print("Parsing answers...")
    answers = parse_answers(html)
    print(f"  Found {len(answers)} answers from page")

    # Build output structure
    output_problems = []
    for p in sorted(problems, key=lambda x: x["num"]):
        n = p["num"]
        answer = answers.get(n)  # May be None if not found on page
        output_problems.append({
            "id": n,
            "problem": p["problem"],
            "answer": answer
        })

    # Validate
    if len(output_problems) != 15:
        print(f"WARNING: Expected 15 problems, found {len(output_problems)}")

    missing_answers = [p["id"] for p in output_problems if p["answer"] is None]
    if missing_answers:
        print(f"WARNING: Missing answers for problems: {missing_answers}")
        print("You will need to manually add answers to data/aime_2026_i.json")

    invalid_answers = [
        p["id"] for p in output_problems
        if p["answer"] is not None and not (0 <= p["answer"] <= 999)
    ]
    if invalid_answers:
        print(f"ERROR: Answers out of range [0,999] for problems: {invalid_answers}")

    result = {
        "exam": "2026_AIME_I",
        "source": URL,
        "problems": output_problems
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved to {OUTPUT}")

    # Summary
    problems_with_answers = sum(1 for p in output_problems if p["answer"] is not None)
    print(f"Problems: {len(output_problems)}/15")
    print(f"Answers:  {problems_with_answers}/15")

    for p in output_problems:
        ans_str = str(p["answer"]) if p["answer"] is not None else "MISSING"
        print(f"  Problem {p['id']:2d}: answer={ans_str}")


if __name__ == "__main__":
    main()
