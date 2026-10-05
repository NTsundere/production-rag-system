@echo off
echo Starting RAG System UI...
python -m streamlit run app/ui.py --server.port 8501
pause