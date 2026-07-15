@echo off
cd C:\Users\pends\Loan
call venv\Scripts\activate.bat
pip install -q pandas numpy scikit-learn streamlit
python model_training.py
pause
