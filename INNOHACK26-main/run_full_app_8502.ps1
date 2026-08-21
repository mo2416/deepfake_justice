$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
python -m streamlit run app.py --server.port 8502 --server.headless true

