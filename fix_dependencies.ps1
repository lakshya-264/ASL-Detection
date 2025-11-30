# PowerShell script to fix dependency compatibility issues
# Run this script to reinstall compatible versions

Write-Host "Fixing dependency compatibility issues..." -ForegroundColor Yellow
Write-Host ""

# Uninstall problematic packages
Write-Host "Uninstalling incompatible packages..." -ForegroundColor Cyan
pip uninstall -y numpy scikit-learn pandas tensorflow

# Install compatible versions
Write-Host ""
Write-Host "Installing compatible versions..." -ForegroundColor Cyan
pip install numpy==1.24.3
pip install scikit-learn==1.3.0
pip install tensorflow==2.13.0
pip install Pillow matplotlib

Write-Host ""
Write-Host "Dependencies fixed! You can now run the training script." -ForegroundColor Green

