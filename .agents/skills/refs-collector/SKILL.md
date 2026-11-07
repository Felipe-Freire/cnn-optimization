---
name: refs-collector
description: >-
  Reads and extracts context from the /refs directory in the workspace. Use when
  the user mentions reference files, physics papers, ATLAS docs, or asks to check
  the /refs folder.
---

# Workspace References Collector

This skill is designed to automatically check the `/refs` folder in the user's project workspace to ground responses in specific domain literature, constraints, or datasets related to the ATLAS HL-LHC project.

## When to Use

- The user asks to summarize reference papers.
- The user needs constraints (e.g., DSP limits, timing constraints) documented in `/refs`.
- Answering questions about calorimeter ring features.

## Steps

1. **Locate Directory**: Look for `./refs` or `/refs` in the current working directory / workspace root.
2. **List Files**: Identify PDFs, Markdown files, or text notes present in the folder.
3. **Extract Context**: Read the relevant sections of the files. If it's a script, analyze the variables. If it's a paper, summarize the requested methodology.
4. **Synthesize**: Answer the user's prompt by directly citing the information found in the `/refs` folder.

## Gotchas

- Do not guess physical constraints; if the user asks for ATLAS specific limits, read them from the `/refs` files first.
- If the directory does not exist or is empty, inform the user to populate it before proceeding.
