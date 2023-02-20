from keras import Sequential
from keras.models import load_model
import cv2
import numpy as np
from keras.utils import img_to_array

model = Sequential()
model_define = load_model('trained_model.h5') 
emotion_types = {0: 'Sad', 1: 'Sad', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Happy'}
types = list(emotion_types.values())
define_faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
define_profiles = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")


"""Authored by: Shadman Mahmood, B00780608 - date modified: 07/21/22
Code description: Used to reduce noise from the input image for better face detection
Strength of median filter is controlled by n parameter which equals to 4"""

def reduce_noise(img, n = 4):
    output = np.zeros(
        (n, n) + (img.shape[0] - n + 1, img.shape[1] - n + 1) + img.shape[2:],
        dtype = img.dtype
    )
    for i in range(n):
        for j in range(n):
            output[i, j] = img[i:i + output.shape[2], j:j + output.shape[3]]
    output = np.moveaxis(output, (0, 1), (-2, -1)).reshape(*output.shape[2:], -1)
    output = np.median(output, axis = -1)
    if output.dtype != img.dtype:
        output = (output.astype(np.float64) + 10 ** -7).astype(img.dtype)
    return output

"""Authored by: Fami Mawla, B00784213 - date modified: 07/21/22
Code description: Performs convolution between 2 2d arrays, such as applying Gaussian filter, etc."""

def convolve2d(img, kernel):
    kernel, output = np.flipud(np.fliplr(kernel)), np.zeros_like(img)
    padded = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
    padded[1:-1, 1:-1] = img
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            output[y, x] = (kernel * padded[y: y + 3, x: x + 3]).sum()
    return output

"""Authored by: Fami Mawla, B00784213 - date modified: 07/21/22
Code description: Converts the image to grayscale by reshaping array to preprocess for other filters, also increases 
contrast for the greyscaled image."""

def grayscale(img):
    vectorised = img[:,:,0]
    vectorised = vectorised.copy()
    vectorised[vectorised < 0] = 0
    return vectorised

"""Authored by: Shadman Mahmood, B00780608 - date modified: 07/21/22
Code description: Captions the emotion captured by the algorithm and displays in final image."""

def output_caption(text, text_x, text_y, img, font_scale = 1, font = cv2.FONT_HERSHEY_DUPLEX, FONT_COLOR = (0, 0, 0), FONT_THICKNESS = 2, rectangle_bgr = (0, 255, 0)):
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=2)[0]
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=2)[0]
    box_coords = ((text_x-10, text_y+4), (text_x + text_width+10, text_y - text_height-5))
    cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(img, text, (text_x, text_y), font, fontScale = font_scale, color = FONT_COLOR, thickness = FONT_THICKNESS)


"""Authored by: Shadman Mahmood, B00780608 - date modified: 07/21/22
Code description: Detects faces, applies convolution to grayscaled image, feature extraction and applies 
rectangular frame to detected faces."""

def detect_faces(img):
    image_gscale = grayscale(img) 
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])    
    image_gscale = convolve2d(image_gscale, kernel)
    faces = define_faces.detectMultiScale(image_gscale, 1.3, 5)
    profiles = define_profiles.detectMultiScale(image_gscale, 1.3, 5)
    if(isinstance(faces, tuple) == False):
        faces = np.array(faces)
    elif(isinstance(profiles,tuple)):
        img= cv2.flip(img, 1)
        image_gscale = grayscale(img) 
        profiles = define_profiles.detectMultiScale(image_gscale, 1.3, 5)
        faces = np.array(profiles)
        print(type(profiles))
    else:
        faces = np.array(profiles)
    faces_detected, rectangle = [], []
    for (x_axis, y_axis, width, height) in faces:
        cv2.rectangle(img, (x_axis, y_axis), (x_axis + width, y_axis + height), (0, 255, 0), 2)
        interest_gscale = image_gscale[y_axis:y_axis + height, x_axis:x_axis + width]
        interest_gscale = cv2.resize(interest_gscale, (48, 48), interpolation = cv2.INTER_AREA)
        faces_detected.append(interest_gscale)
        rectangle.append((x_axis, width, y_axis, height))
    return rectangle, faces_detected, img


"""Authored by: Fami Mawla, B00784213 - date modified: 07/21/22
Code description: Reduces noise, contains algorithm to predict expressions from the faces detected using the
trained model dataset, also outputs the final image with captioned emotion."""

def detect_emotions(img):
    img = cv2.imread(img)
    img = reduce_noise(img)
    rectangle, faces, image = detect_faces(img)
    i = 0
    for expression in faces:
        interest = expression.astype("float") / 255.0
        interest = img_to_array(interest)
        interest = np. expand_dims(interest, axis = 0)
        predictions = model_define.predict(interest)[0]
        label, label_position = emotion_types[predictions.argmax()], (rectangle[i][0] + int((rectangle[i][1] / 2)), abs(rectangle[i][2] - 10))
        i =+ 1
        output_caption(label, label_position[0],label_position[1], image)
    filename = 'savedImage.jpg'
    cv2.imwrite(filename, img)
    cv2.imshow("Emotion Recognition", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#main function
if __name__ == '__main__':
    print("Emotion Recognition Software \nTo exit enter 'q' at anytime.")
    while True:
        USER_INPUT = input("Please enter image filename:")
        if(USER_INPUT.lower() == "q"):
            break
        detect_emotions(USER_INPUT)    