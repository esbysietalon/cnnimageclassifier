@echo Installing python dependencies..

@echo off
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
mkdir bucket >nul 2>&1
cd bucket
mkdir abnormal >nul 2>&1
mkdir normal >nul 2>&1
cd ..
@echo Creating start.bat..
del start.bat >nul 2>&1
@echo @echo off >> start.bat
@echo TITLE CNN IMAGE CLASSIFIER v0.2.10 >> start.bat
@echo python cnn.py >> start.bat
@echo Installation complete.
