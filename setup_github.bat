@echo off
REM ==============================================
REM  GitHub Repository Setup Script
REM  For Windows
REM ==============================================

echo.
echo ============================================
echo   GitHub Setup for Your Project
echo ============================================
echo.

REM --- Check if Git is installed ---
echo [1/4] Checking Git installation...
git --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Git is not installed!
    echo.
    echo Please download and install Git from:
    echo   https://git-scm.com/download/windows
    echo.
    echo After installing, restart this script.
    echo.
    pause
    exit /b 1
)
git --version
echo Git found!
echo.

REM --- Configure Git (if not already configured) ---
echo [2/4] Checking Git configuration...
git config --global user.name >nul 2>&1
if errorlevel 1 (
    echo.
    echo Git needs to be configured with your name and email.
    echo.
    set /p GIT_NAME="Enter your name (for commits): "
    set /p GIT_EMAIL="Enter your email (same as GitHub): "
    git config --global user.name "%GIT_NAME%"
    git config --global user.email "%GIT_EMAIL%"
    echo Configuration saved!
) else (
    echo Git is already configured:
    echo   Name:  
    git config --global user.name
    echo   Email: 
    git config --global user.email
)
echo.

REM --- Initialize Git repository ---
echo [3/4] Initializing Git repository...
if exist ".git" (
    echo Git repository already exists.
) else (
    git init
    echo Git repository initialized!
)
echo.

REM --- Create initial commit ---
echo [4/4] Creating initial commit...
git add .
git commit -m "Initial commit - Project setup"
echo.

echo ============================================
echo   LOCAL GIT SETUP COMPLETE!
echo ============================================
echo.
echo NEXT STEPS - Connect to GitHub:
echo.
echo 1. Go to https://github.com and log in
echo 2. Click the "+" icon (top right) and select "New repository"
echo 3. Enter a repository name (e.g., ml_project01)
echo 4. Do NOT initialize with README (you already have files)
echo 5. Click "Create repository"
echo 6. Copy the commands from "push an existing repository"
echo.
echo Or run these commands (replace YOUR_USERNAME and REPO_NAME):
echo.
echo   git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
echo   git branch -M main
echo   git push -u origin main
echo.
echo ============================================
echo.

pause

