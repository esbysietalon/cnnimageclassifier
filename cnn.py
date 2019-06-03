'''
Created on Nov 24, 2018

@author: Joaquin
'''


print("Starting up..")

from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from pathlib import Path
from random import SystemRandom
import os
import traceback

tutorial = []
tutorial.append("""Getting Started.
Welcome to CNN Image Classifier.
If you have not used this before, or you're not sure how to use this tool, please see the following instructions:
You can also view this and other tutorial pages by navigating to where cnn.py is and navigating to the tutorial folder.
This page is 'tutorial0.txt'.
\n
1.  Enter '!help' to view a list of commands. You can do this if you forget a command.
2.  Enter '!configure' to configure training. Here is where you will set the steps_per_epoch, number of epochs,
    and number of validation_steps
    To begin with, try setting the steps_per_epoch to 100, number of epochs to 10, and number of validation_steps to 25.
    (Enter 100, then enter 10, then enter 25)
    Please note that the lower these numbers, the faster training will take, but the less accurate the model will be.
3.  Enter '!train' to begin training the model. You should see the program start printing something like:
\n
    1/8000 [..............................] - ETA 15:00 - loss: 0.0448 - acc: 0.9845 - val_loss: 2.2941 - val_acc: 0.6905
\n
    (Your numbers will differ)
\n
    Take special notice of the value in front of 'acc:', in this case 0.9845. This is the accuracy of the model.
    Training increases the accuracy of the model slowly, the more time you give it, the more accurate it will become.
    Until accuracy is very high (0.999+), the model can be very unreliable when classifying similar images.
    This can take dozens of hours worth of training depending on how fast this machine is (25+).
    However, this training time can be split up over multiple sessions as you will see.
4.  After training is complete, enter '!save', then select a filename to save the model to (ex: 'model01.h5')
    This is what allows us to split up hours and hours of training time over manageable sessions.
5.  Now restart the program. Exit the program by entering '!exit' or closing the window.
    Start it again by clicking 'start.bat'. Enter '!tutorial 0' to view this page again.
6.  You may notice that you get a warning saying that the model is untrained. Every time the program starts up
    it starts with an untrained model. We can fix this by entering '!load', then entering the filename that you
    chose earlier (ex: 'model01.h5').
7.  Now we can either continue training (recommended) or start classifying images. I recommend that you repeat steps 8-11
    until the model has an accuracy of at least 0.999.
8.  Enter '!load', then enter the filename of the model you're currently training. In this tutorial that was 'model01.h5'
9.  Enter '!configure' and select values that you think are reasonable
    (I recommend you numbers in the range of steps_per_epoch = 8000, 2 < epochs < 5,
    and validation_steps = 2000).
    Higher values mean more training, which means more accuracy, but longer training sessions.
10. Enter '!train' and wait for training to complete.
11. Enter '!save' then enter the desired filename you want to save the model as.
12. Once the model you're training has an accuracy of at least 0.997, move on to the next part of the tutorial,
    which can be done by entering '!tutorial 1' or navigating to where cnn.py is and navigating to the tutorial
    folder and opening 'tutorial1.txt'.  
""")
tutorial.append("""Using the Classifier
You can view this and other tutorial pages by navigating to where cnn.py is and navigating to the tutorial folder.
This page is 'tutorial1.txt'.
Please do not continue if the model you currently have does not have an accuracy of at least 0.999.
You can verify the accuracy of the model by loading it then training it briefly to look at the value of 'acc:'.
The following instructions will walk you through how to use the classifier to classify images.
\n
1.   Switch to classifier mode by entering '!classify'. You cannot use any other commands (!help, !load, !train, etc)
     while in classifier mode. You can exit classifier mode by entering '!exit'.
2.   Enter the filepath to a .jpg file you wish to classify. The classifier will tell you if it thinks it is NORMAL
     or ABNORMAL.
3.   To classify multiple images, enter their names separated by a space. For example, to classify xray.jpg and
     xray2.jpg, enter 'xray.jpg xray2.jpg'.
4.   You can also classify all .jpg files in the folder where cnn.py is by entering '*.jpg'. Try this now.
5.   You can also classify all .jpg files that start with 'xr' by entering 'xr*.jpg'. Try this now.
     You can use this with any parts of any .jpg files (ex: you can classify 'firsttest.jpg' 'secondtest.jpg' and
     'thirdtest.jpg' by entering '*test.jpg').
6.   You can also classify all .jpg files in a folder by entering 'foldername/*.jpg' (ex: to classify all .jpg files
     in the folder 'xrays', enter 'xrays/*.jpg').
7.   Note that the lists of abnormal and normal images can be found in the results/ folder.
""")

def printHelp():
    print("Please see below a list of commands: ")
    print("-------------------------------------------")
    print("!tutorial - use this command if you're lost")
    print("!load - loads a model from a .h5 file")
    print("!save - saves a model as a .h5 file")
    print("!reset - deletes the existing model")
    print("!clear - clears the lists of normal/abnormal images")
    print("!help - displays a list of commands")
    print("!classify - switches to classifying mode")
    print("!addimages - adds images from bucket, portioned between test and training sets")
    print("!checkmodel - checks accuracy of model on images in test_set/")
    print("!exit - exits classifying mode or program")
    print("!configure - configures training")
    print("!train - trains model")
    print("-------------------------------------------")
def printTutorial(page):
    if page >= 0 and page < tutorial.__len__():
        print(tutorial[page])
    else:
        print("Table of Contents\n")
        print("Access a page of the tutorial by entering '!tutorial #', where # is the page you want to access.")
        print("You can also access the tutorial pages by going to the tutorial folder where cnn.py is installed.")
        print("0 - Getting Started")
        print("1 - Using the Classifier")
def printIntro():
    print("===========================================")
    print("       CNN IMAGE CLASSIFIER v0.2.8")
    print("===========================================")
def openListFiles():
    normalList = open("results/normallist.txt", "w+")
    normalList.close()
    abnormalList = open("results/abnormallist.txt", "w+")
    abnormalList.close()
def portionBucket():
    cryptorand = SystemRandom()
    print("Portioning images from bucket..")
    if len(os.listdir("bucket/normal/")) > 0:
        print("Portioning normal images..")
        transferlist = list(Path('.').glob("bucket/normal/*.jpg"))
        cryptorand.shuffle(transferlist)
        portionlimit = transferlist.__len__() / 3
        index = 0
        for s in transferlist:
            index += 1
            if index < portionlimit:
                os.rename(s,"internal/test_set/normal/"+str(s)[14:])
            else:
                os.rename(s,"internal/training_set/normal/"+str(s)[14:])
    else:
        print("No normal images found.")
    if len(os.listdir("bucket/abnormal/")) > 0:
        print("Portioning abnormal images..")
        transferlist = list(Path('.').glob("bucket/abnormal/*.jpg"))
        cryptorand.shuffle(transferlist)
        portionlimit = transferlist.__len__() / 3
        index = 0
        for s in transferlist:
            index += 1
            if index < portionlimit:
                os.rename(s,"internal/test_set/abnormal/"+str(s)[15:])
            else:
                os.rename(s,"internal/training_set/abnormal/"+str(s)[15:])
    else:
        print("No abnormal images found.")
    print("Images portioned.")
     

def configureTraining(variable, currval):
    if variable.casefold() == "steps_per_epoch":
        input_steps = 0
        while(input_steps < 1):
            print("Please input desired number of steps (default: 8000, current: " + str(currval) + "):")
            print(">", end="")
            try:
                input_steps = int(input())
            except Exception as ex:
                traceback.print_exc()
                print("Configure failed, using default value")
                input_steps = 8000
        return input_steps
    if variable.casefold() == "epochs":       
        input_epochs = 0
        while(input_epochs < 1):
            print("Please input desired number of epochs (default: 25, current: " + str(currval) + "):")
            print(">", end="")
            try:
                input_epochs = int(input())
            except Exception as ex:
                traceback.print_exc()
                print("Configure failed, using default value")
                input_epochs = 25
        return input_epochs
    if variable.casefold() == "validation_steps":
        input_validation_steps = 0;
        while(input_validation_steps < 1):
            print("Please input desired number of validation steps (default: 2000, current: " + str(currval) + "):")
            print(">", end="")
            try:
                input_validation_steps = int(input())
            except Exception as ex:
                traceback.print_exc()
                print("Configure failed, using default value")
                input_validation_steps = 2000
        return input_validation_steps
    print("Variable not found. Available variables to configure are:")
    print("steps_per_epoch - this determines the steps per epoch (default: 8000)")
    print("epochs - this determines the number of epochs/generations (default: 25)")
    print("validation_steps - this determines the steps to validate at the end of each epoch (default: 2000)")


portionBucket()


classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 128, activation='relu'))
classifier.add(Dense(units = 1, activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('internal/training_set', target_size=(64, 64), batch_size=32, class_mode='binary')
test_set = test_datagen.flow_from_directory('internal/test_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')



steps_per_epoch = 8000
epochs = 25
validation_steps = 2000    
rawModel = True
classify = False

import numpy as np
from keras.preprocessing import image



openListFiles()

if not os.path.exists("tutorial"):
    os.mkdir("tutorial")

for i in range(0, tutorial.__len__()):
    if not os.path.exists("tutorial/tutorial"+str(i)+".txt"):
        tutfile = open("tutorial/tutorial"+str(i)+".txt", "w+")
        tutfile.write(tutorial[i])
        tutfile.close()

printIntro()
printHelp()
print(" ")
        
while 1==1:
    if rawModel:
        print("Model is untrained. Recommended: !load or !configure then !train")
    if classify:
        print("Enter the filepath/s to the image (!exit to exit classifying mode):")
    else:
        print("Please type in a command (type !help to see a list of commands):") 
    
    print(">", end="")
    #if not classify:
    #    user_input = str(input()).split()[0]
    #else:
    list_uinput = str(input()).split()
    user_input = list_uinput[0]
    
    if user_input == "!exit":
        if classify:
            classify = False
            print("Exited classifying mode.")
        else:
            break
    if classify:
        if user_input.startswith('!'):
            print("Use !exit to exit classifier mode.")
        print(" ")
        print("Found the following files to classify:")
        filecount = 0
        for uinput in list_uinput:
            globlist = Path('.').glob(uinput)
            for ginput in globlist:
                print(ginput)
                filecount+=1
        if filecount == 0:
            print("No files found. Please check that filepath is correct.")
        print(" ")
        currcount = 0
        abnormalCount = 0
        normalCount = 0
        abnormalFiles = []
        normalFiles = []
        for uinput in list_uinput:
            globlist = Path('.').glob(uinput)
            for ginput in globlist:
                ginput = str(ginput)
                currcount += 1
                try:
                    test_image = image.load_img(ginput, target_size = (64, 64))
                    test_image = image.img_to_array(test_image)
                    test_image = np.expand_dims(test_image, axis = 0)
                    result = classifier.predict(test_image)
                    training_set.class_indices
                    if result[0][0] >= 0.5:
                        prediction = ginput + ': NORMAL ' + '(' + str(result[0][0]) + ')'
                        normalFiles.append(ginput)
                        normalCount += 1
                    else:
                        prediction = ginput + ': ABNORMAL ' + '(' + str(result[0][0]) + ')'
                        abnormalFiles.append(ginput)
                        abnormalCount += 1
                    print("["+str(currcount) + "/" + str(filecount) + "] " + prediction)
                except Exception as ex:
                    print("["+str(currcount) + "/" + str(filecount) + "] " + ginput + ": error encountered. Invalid file chosen.")
        normalList = open("results/normallist.txt", "a")
        for file in normalFiles:
            normalList.write(file + '\n')
        normalList.close()
        abnormalList = open("results/abnormallist.txt", "a")
        for file in abnormalFiles:
            abnormalList.write(file + '\n')
        abnormalList.close()
        print(str(normalCount) + " normal images found. " + str(abnormalCount) + " abnormal images found. See abnormallist.txt and normallist.txt in the results/ folder for lists of normal and abnormal images respectively.")
    else:
        if user_input == "!addimages":
            portionBucket()
        if user_input == "!tutorial":
            if list_uinput.__len__() > 1:
                try:
                    page = int(list_uinput[1])
                    printTutorial(page)
                except Exception as ex:
                    print("Use /'!tutorial #/' to access a specific page of the tutorial (where # is the page you want to access).")
            else:
                printTutorial(-1)
        if user_input == "!help":
            printHelp()
        if user_input == "!clear":
            print("Are you sure you want to clear the lists of normal/abnormal images? (y/n)")
            print(">", end="")
            user_input = str(input()).split()[0]
            if user_input.casefold() == "y" or user_input.casefold() == "yes":
                openListFiles()
                print("Lists cleared.")
            else:
                print("Did not clear lists.")
        if user_input == "!save":
            forceSave = False
            save_file = ""
            if list_uinput.__len__() > 1:
                save_file = list_uinput[1]
            else:
                save_file = ""
            if rawModel:
                print("Warning: Model is untrained. Are you sure you want to save it anyway? (y/n)")
                print(">", end="")
                user_input = str(input()).split()[0]
                if user_input.casefold() == "y" or user_input.casefold() == "yes":
                    forceSave = True
            if rawModel and not forceSave:
                print("Save aborted.")
            if not rawModel or forceSave:
                if save_file == "":
                    print("What should the file be called?")
                    print(">", end="")
                    save_file = str(input()).split()[0]
                try:
                    classifier.save(save_file)
                    print("Save complete.")
                except Exception as ex:
                    print("Save failed!")
                    traceback.print_exc()
        if user_input == "!load":
            load_file = ""
            if list_uinput.__len__() > 1:
                load_file = list_uinput[1]
            else:
                load_file = ""
            if load_file == "":
                print("What file should the model be loaded from? (must be a .h5 file)")
                print(">", end="")
                load_file = str(input()).split()[0]
            try:
                classifier = load_model(load_file)
                rawModel = False
                print("Load complete.")
            except Exception as ex:
                print("Load failed!")
                traceback.print_exc()
        if user_input == "!reset":
            print("Are you sure you want to delete the existing model? (y/n)")
            print(">", end="")
            user_input = str(input()).split()[0]
            if user_input.casefold() == "y" or user_input.casefold() == "yes":
                rawModel = True
                classifier = Sequential()
                classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
                classifier.add(MaxPooling2D(pool_size = (2, 2)))
                classifier.add(Flatten())
                classifier.add(Dense(units = 128, activation='relu'))
                classifier.add(Dense(units = 1, activation='sigmoid'))
                classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                print("Model reset.")
            else:
                print("Reset aborted.")
        if user_input == "!configure":
            if list_uinput.__len__() > 1:
                if list_uinput[1] == "steps_per_epoch":
                    steps_per_epoch = configureTraining("steps_per_epoch", steps_per_epoch)
                elif list_uinput[1] == "epochs":
                    epochs = configureTraining("epochs", epochs)
                elif list_uinput[1] == "validation_steps":
                    validation_steps = configureTraining("validation_steps", validation_steps)
                else:
                    configureTraining(list_uinput[1], -1)
            else:
                steps_per_epoch = configureTraining("steps_per_epoch", steps_per_epoch)
                epochs = configureTraining("epochs", epochs)
                validation_steps = configureTraining("validation_steps", validation_steps)
        if user_input == "!train":
            confirmSettings = False
            while not confirmSettings:
                print("Current configuration settings:")
                print("Steps per epoch: " + str(steps_per_epoch))
                print("Epochs: " + str(epochs))
                print("Validation steps: " + str(validation_steps))
                print("Is this okay? (y/n/cancel)")
                print(">", end="")
                user_input = str(input()).split()[0]
                if user_input.casefold() == "y" or user_input.casefold() == "yes":
                    confirmSettings = True
                    classifier.fit_generator(training_set, steps_per_epoch, epochs, validation_data = test_set, validation_steps = validation_steps)
                    print("Training complete.")
                    rawModel = False
                elif user_input.casefold() == "n" or user_input.casefold() == "no":
                    steps_per_epoch = configureTraining("steps_per_epoch", steps_per_epoch)
                    epochs = configureTraining("epochs", epochs)
                    validation_steps = configureTraining("validation_steps", validation_steps)
                else:
                    print("Training aborted.")
                    break
        if user_input == "!classify":
            forceClassify = False
            if rawModel:
                print("Warning: Model is untrained. Are you sure you want to enter classifying mode anyway? (y/n)")
                print(">", end="")
                user_input = str(input()).split()[0]
                if user_input.casefold() == "y" or user_input.casefold() == "yes":
                    forceClassify = True
            if rawModel and not forceClassify:
                print("Did not enter classifying mode.")
            if not rawModel or forceClassify:
                classify = True
                print("Entered classifying mode.")
        if user_input == "!checkmodel":
            print("Checking model on test_set..")
            currcount = 0
            correct_total = 0
            false_negatives = 0
            positives = 0
            false_positives = 0
            negatives = 0
            filecount = list(Path('.').glob("internal/test_set/*/*.jpg")).__len__()
            if filecount == 0:
                print("No test images found." 
                print("Recommended: add images into bucket/abnormal/ and bucket/normal/ then use !addimages")
            globlist = Path('.').glob("internal/test_set/abnormal/*.jpg")
            for ginput in globlist:
                ginput = str(ginput)
                try:
                    currcount += 1
                    positives += 1
                    test_image = image.load_img(ginput, target_size = (64, 64))
                    test_image = image.img_to_array(test_image)
                    test_image = np.expand_dims(test_image, axis = 0)
                    result = classifier.predict(test_image)
                    training_set.class_indices
                    if result[0][0] >= 0.5:
                        prediction = ginput + ': NORMAL ' + '(' + str(result[0][0]) + ')'
                        false_negatives += 1
                    else:
                        prediction = ginput + ': ABNORMAL ' + '(' + str(result[0][0]) + ')'
                        correct_total += 1
                    print("["+str(currcount) + "/" + str(filecount) + "] " + prediction)
                except Exception as ex:
                    print("["+str(currcount) + "/" + str(filecount) + "] " + ginput + ": error encountered. Invalid file chosen.")
            globlist = Path('.').glob("internal/test_set/normal/*.jpg")
            for ginput in globlist:
                ginput = str(ginput)
                try:
                    currcount += 1
                    negatives += 1
                    test_image = image.load_img(ginput, target_size = (64, 64))
                    test_image = image.img_to_array(test_image)
                    test_image = np.expand_dims(test_image, axis = 0)
                    result = classifier.predict(test_image)
                    training_set.class_indices
                    if result[0][0] >= 0.5:
                        prediction = ginput + ': NORMAL ' + '(' + str(result[0][0]) + ')'
                        correct_total += 1
                    else:
                        prediction = ginput + ': ABNORMAL ' + '(' + str(result[0][0]) + ')'
                        false_positives += 1
                    print("["+str(currcount) + "/" + str(filecount) + "] " + prediction)
                except Exception as ex:
                    print("["+str(currcount) + "/" + str(filecount) + "] " + ginput + ": error encountered. Invalid file chosen.")
            if currcount > 0:
                print("Findings..")
                print("Total Accuracy: " + str(float(correct_total)/float(currcount)) + " (" + str(correct_total) + "/" + str(currcount) + ")")
                if negatives > 0:
                    print("False Positives (Normal images classified as Abnormal): " + str(float(false_positives) / float(negatives)) + " (" + str(false_positives) + "/" + str(negatives) + ")")
                if positives > 0:
                    print("False Negatives (Abnormal images classified as Normal): " + str(float(false_negatives) / float(positives)) + " (" + str(false_negatives) + "/" + str(positives) + ")")
                print("Conclusion:")
                if float(correct_total)/float(currcount) < 0.95:
                    print("Requires more training! Try using larger configuration numbers.")
                else:
                    print("May be used for classification, keeping in mind error margins.")
    print(" ")