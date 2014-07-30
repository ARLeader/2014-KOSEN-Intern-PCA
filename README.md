2014-KOSEN-Intern-PCA
=====================

Kuroki Lab Internship PCA Lab

Requirements
-----------

- OpenCV Version 2.4.9
- Python Version 2.7.8

The application is meant to run on Release Build

We have prepared a Property Sheet in the Release Repository

Preparing files and folders
-------------------

The folder creation commands has not been implemented yet, The user have to manually prepare the appropriate folder or the application may not be able to write files

Please prepare the following folders on the following directories.

> - Camera captures directory : C:\camface\captures
- Images database directory : C:\camface\db
- (Optional) Output Matrix directory : C:\faceout
- (Incomplete)(Optional) Python script directory : C:\camface\db 

Configuring the environment


Capturing Faces using the application
----------------

After running, the application will immidiately try to read the files. If the CSV files is not detected, the frame displayed on the screen will be labeled as 'Unknown'

###Keyboard inputs

> - Escape : Quit the application
- N : The application will ask for the name of the detected face, and start capture them continuously for 40 images
- I : The application will ask for the name of the detected face, pressing a key will make the application capture an image until user pressed T

The files will be written down at the camera captures directory. The user will have to manually move the file to the images database directory and create folders for each person recorded.

Example of file structures

> C:\camface\db\01
>
> C:\camface\db\02
>
> C:\camface\db\03
>
>.
>.
>.

Running the Python Script
=========================

The script create an appropriate CSV file named Facedir.txt on the Image database directory. The script have the following parameters

- create_csv.py <Image Database Directory>
