@echo off
echo Fixing dependency compatibility issues...
echo.

echo Uninstalling incompatible packages...
pip uninstall -y numpy scikit-learn pandas tensorflow

echo.
echo Installing compatible versions...
pip install numpy==1.24.3
pip install scikit-learn==1.3.0
pip install tensorflow==2.13.0
pip install Pillow matplotlib

echo.
echo Dependencies fixed! You can now run the training script.
pause

