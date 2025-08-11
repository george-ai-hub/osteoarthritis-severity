@echo off
echo.
echo Osteoarthritis Clinical Decision Support System
echo =================================================
echo.
echo Starting clinical application...
echo This will open your web browser to http://localhost:8501
echo.

REM Activate conda base environment and run streamlit
echo Starting standalone application...
call conda activate base
python -m streamlit run clinical_app_standalone.py --server.port=8501

if %errorlevel% neq 0 (
    echo.
    echo WARNING: Could not start the application
    echo.
    echo Please ensure:
    echo   - Python and Streamlit are installed
    echo   - You are in the correct directory
    echo   - Conda environment is properly set up
    echo.
    echo Manual launch command:
    echo   python -m streamlit run clinical_app_standalone.py
    echo.
    echo Press any key to close...
    pause >nul
) else (
    echo.
    echo Application started successfully!
    echo Press any key to close this window...
    pause >nul
) 