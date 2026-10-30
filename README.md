# Agentic Data Analyst

A practical MVP for CSV exploration using Python, Pandas, Streamlit, Plotly and scikit-learn. An optional OpenAI API key adds an explanation layer grounded in computed results.

## Features
- CSV upload and preview
- Automated profile: data types, missing values, unique counts, duplicates and descriptive statistics
- Natural-language question router for common analytical questions
- Interactive Plotly charts
- Optional LLM explanation based on computed evidence
- Baseline Random Forest classification/regression with holdout metrics
- LangGraph dependency included for extending the workflow into a graph-based agent

## Run in VS Code (Windows)
1. Extract the ZIP and open the `agentic_data_analyst` folder in VS Code.
2. Open Terminal > New Terminal.
3. Create a virtual environment:
   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, run:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .venv\Scripts\Activate.ps1
   ```
4. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. Launch:
   ```powershell
   streamlit run app.py
   ```
6. Browser opens locally. Upload `sample_sales.csv` or your own CSV.

## Optional LLM explanations
Paste an OpenAI API key into the app's sidebar. Without a key, data profiling, supported question analysis, charts and baseline ML evaluation still work. Never commit API keys to GitHub.

## Demo questions
- Which category has the highest average sales?
- Are there any missing values?
- Show the distribution of sales.
- What are the correlations between numeric columns?
- Give me summary statistics.
- How many duplicate rows are there?

## Honest project status
This is a working portfolio MVP, not a production-grade autonomous analyst. The natural-language router supports common question patterns. Validate outputs on your own dataset and only claim features you have tested. LangGraph is included as a dependency to support a graph-based workflow extension; the current MVP's core routing is deterministic.
