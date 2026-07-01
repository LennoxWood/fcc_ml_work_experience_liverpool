@echo off
:: setup.bat
::
:: FCC HH Analysis -- Windows setup script
::
:: REQUIREMENTS: Git for Windows must be installed (https://git-scm.com/download/win)
:: Git for Windows includes Git Bash, which this script uses to run the setup.
::
:: USAGE: Double-click this file, or run it from a terminal:
::   setup.bat
::
:: First run: installs conda and creates the Python environment (~10-15 mins).
:: Subsequent runs: activates the environment and opens JupyterLab directly.

echo ============================================================
echo   FCC HH Analysis -- Environment Setup
echo ============================================================
echo.

:: Check Git Bash is available
where bash >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Git Bash not found.
    echo.
    echo Please install Git for Windows from:
    echo   https://git-scm.com/download/win
    echo.
    echo Make sure to tick "Add Git Bash to PATH" during installation,
    echo then close and reopen this window.
    pause
    exit /b 1
)

:: Run the bash setup script through Git Bash
:: The script lives in the same directory as this .bat file
bash -c "source \"%~dp0FCCsetup.sh\""

pause
