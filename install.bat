pip install Keras
pip install Theano
pip install tensorflow
pip install pathlib

mkdir results
mkdir internal
cd internal
mkdir test_set
cd test_set
mkdir normal
mkdir abnormal
cd ..
mkdir training_set
cd training_set
mkdir normal
mkdir abnormal
cd ..
cd ..
mkdir bucket
cd bucket
mkdir abnormal
mkdir normal
cd ..

echo @echo off >> start.bat
echo TITLE CNN IMAGE CLASSIFIER v0.2.7 >> start.bat
echo python cnn.py >> start.bat
