@echo off
echo === Installing Python Packages for Resume RAG System ===
echo.

echo Updating pip...
python -m pip install --upgrade pip

echo.
echo Installing packages...
python -m pip install sentence-transformers numpy openai PyPDF2

echo.
echo Verifying installations...
python -c "import sentence_transformers; print('✓ sentence-transformers installed')"
python -c "import numpy; print('✓ numpy installed')"
python -c "import openai; print('✓ openai installed')"
python -c "import PyPDF2; print('✓ PyPDF2 installed')"

echo.
echo Setup complete!
pause