#!/usr/bin/env python3
"""
generate_code_paper.py

Walk a folder tree, pick up source files, and generate a LaTeX paper
with each file included via the `listings` package. Supports excluding
files/directories via regex patterns. Compile with xelatex for emoji/symbol support.
"""

import os
import re
import argparse
from datetime import datetime

# Default regex patterns to exclude
DEFAULT_EXCLUDE_PATTERNS = [
    r"__pycache__",
    r"\.git",
    r"\.idea",
    r"\.vscode",
    r"\.env",
    r"venv",
    r"node_modules",
    r"\.venv",
    r"\.pyc$",
    r"\.DS_Store$",
]

# Default file extensions to include
DEFAULT_EXTENSIONS = [".py", ".js", ".java", ".c", ".cpp", ".hs", ".rs"]

TEX_POSTAMBLE = r"""\end{document}
"""


def build_preamble(date_str):
    return rf"""\documentclass[12pt]{{article}}
\usepackage{{fontspec}}                % XeLaTeX for full Unicode
\usepackage{{emoji}}                   % simple emoji support
\usepackage{{listings}}
\usepackage{{xcolor}}
\lstset{{
  basicstyle=\ttfamily\small,
  breaklines=true,
  keywordstyle=\color{{blue}},
  commentstyle=\color{{gray}},
  stringstyle=\color{{olive}},
  showstringspaces=false,
  upquote=true
}}
\title{{Collected Code Listing}}
\author{{Generated on {date_str}}}
\date{{}}
\begin{{document}}
\maketitle
\tableofcontents
\bigskip
"""


def should_exclude(path, patterns):
    return any(pattern.search(path) for pattern in patterns)


def find_code_files(root, exclude_patterns, include_extensions):
    exclude_regex = [re.compile(p) for p in exclude_patterns]
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d
            for d in dirnames
            if not should_exclude(
                os.path.relpath(os.path.join(dirpath, d), root), exclude_regex
            )
        ]
        for fname in filenames:
            rel_path = os.path.relpath(os.path.join(dirpath, fname), root)
            if should_exclude(rel_path, exclude_regex):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if include_extensions and ext not in include_extensions:
                continue
            yield rel_path


def make_tex_document(root, files, outpath):
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    preamble = build_preamble(date_str)
    with open(outpath, "w", encoding="utf-8") as tex:
        tex.write(preamble)
        for rel in files:
            section = rel.replace(os.sep, "/")
            tex.write(f"\\section{{{section}}}\n")
            tex.write(
                f"\\lstinputlisting[caption={{{section}}}]{{{os.path.join(root, rel).replace(os.sep, '/')}}}\n\n"
            )
        tex.write(TEX_POSTAMBLE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX paper embedding all code files from a folder."
    )
    parser.add_argument("input_dir", help="Root directory to scan for code files")
    parser.add_argument(
        "-o", "--output", default="code_paper.tex", help="Output .tex file path"
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=DEFAULT_EXCLUDE_PATTERNS,
        help="Regex patterns to exclude files/directories",
    )
    parser.add_argument(
        "--ext",
        nargs="+",
        default=DEFAULT_EXTENSIONS,
        help="File extensions to include (e.g. .py .js)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.input_dir
    exclude_patterns = args.exclude
    include_extensions = [e if e.startswith(".") else f".{e}" for e in args.ext]

    files = sorted(find_code_files(root, exclude_patterns, include_extensions))
    if not files:
        print("No code files found. Check your exclude patterns or extensions.")
        return

    print(f"Found {len(files)} files; writing LaTeX to {args.output}")
    make_tex_document(root, files, args.output)
    print("Done! Compile with: xelatex -shell-escape", args.output)


if __name__ == "__main__":
    main()
