$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
python -m streamlit run tester_app.py --server.port 4747 --server.headless true

