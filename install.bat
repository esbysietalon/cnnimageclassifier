@echo Checking if Python is installed..
@echo off

reg query "hkcu\software\Python\PythonCore\3.6" >NUL
if ERRORLEVEL 1 GOTO NOPYTHON
goto :HASPYTHON

:NOPYTHON

@echo Installing Python (3.6.0, 32-bit)..
cd dependency
python-3.6.0.exe
cd ..
reg query "hkcu\software\Python\PythonCore\3.6"
if ERRORLEVEL 1 exit

:HASPYTHON

@echo Installing Python dependencies..

pip install Keras
pip install Theano
pip install tensorflow
pip install pathlib
@echo Creating folder structures..
mkdir results >nul 2>&1
mkdir internal >nul 2>&1
cd internal
mkdir logs >nul 2>&1
mkdir tmp >nul 2>&1
cd tmp
mkdir checkpoints >nul 2>&1
mkdir tensorboard >nul 2>&1
cd ..
mkdir test_set >nul 2>&1
cd test_set
mkdir normal >nul 2>&1
mkdir abnormal >nul 2>&1
cd ..
mkdir training_set >nul 2>&1
cd training_set
mkdir normal >nul 2>&1
mkdir abnormal >nul 2>&1
cd ..
cd ..
mkdir PLACE_IMAGES_HERE >nul 2>&1
cd PLACE_IMAGES_HERE
mkdir abnormal >nul 2>&1
mkdir normal >nul 2>&1
cd ..
@echo Creating start.bat..
del start.bat >nul 2>&1
@echo @echo off >> start.bat
@echo TITLE CNN IMAGE CLASSIFIER v0.2.11 >> start.bat
@echo python cnn.py >> start.bat
@echo Installation complete.
