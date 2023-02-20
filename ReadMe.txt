CONTENTS OF THIS FILE
---------------------
 * Overview
 * Dependencies
 * GitLab link

# Note: Some sample images have been provided

Overview
------------

Project    :        Emotion recognition filter 
Authors    :        Shadman Mahmood
Constraints:        Certain side profile emotions cannot be detected. OpenCV constraints on detecting certain faces.
Output     :        Produces detected emotion as a caption on the image which is saved as a new file savedImg.jpg
Bugs-fixed :        OpenCV can only detect certain left-sided profiles so images as flipped if they are right sided to perform detection, 
                    slightly tilted faces can now be used in the program to detect emotion. Check sample(happyside1.png,happyside2.png,sad2.png)

Dependencies
------------

Libraries           :       Keras, TensorFlow, OpenCV, NumPy
Prediction Model    :       trained_model.h5 (this is supplied in the zip folder)



