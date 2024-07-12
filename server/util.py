# Import necessary libraries
import joblib
import json
import numpy as np
import base64
import cv2
from wavelet import w2d

# Initialize dictionaries to store class names and numbers
__class_name_to_number = {}
__class_number_to_name = {}

# Initialize the model variable
__model = None

# Function to classify an image
def classify_image(image_base64_data, file_path=None):
    # Get the cropped image(s) from the input image
    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)

    # Initialize an empty list to store the results
    result = []

    # Loop through each cropped image
    for img in imgs:
        # Resize the image to 32x32
        scalled_raw_img = cv2.resize(img, (32, 32))

        # Apply wavelet transform to the image
        img_har = w2d(img, 'db1', 5)

        # Resize the wavelet transformed image to 32x32
        scalled_img_har = cv2.resize(img_har, (32, 32))

        # Combine the original and wavelet transformed images
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))

        # Reshape the combined image to a 1D array
        len_image_array = 32*32*3 + 32*32
        final = combined_img.reshape(1,len_image_array).astype(float)

        # Use the model to predict the class of the image
        prediction = __model.predict(final)[0]

        # Get the class name from the class number
        class_name = class_number_to_name(prediction)

        # Get the class probabilities
        class_probabilities = np.around(__model.predict_proba(final)*100,2).tolist()[0]

        # Store the result in a dictionary
        result.append({
            'class': class_name,
            'class_probability': class_probabilities,
            'class_dictionary': __class_name_to_number
        })

    # Return the result
    return result

# Function to convert a class number to a class name
def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

# Function to load saved artifacts (model and class dictionaries)
def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    # Load the class dictionary from a JSON file
    with open("./server/artifacts/class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

    # Load the model from a pickle file
    global __model
    if __model is None:
        with open('./server/artifacts/save_model.pkl', 'rb') as f:
            __model = joblib.load(f)
    print("loading saved artifacts...done")

# Function to convert a base64 encoded string to a CV2 image
def get_cv2_image_from_base64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    '''
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

# Function to get the cropped image(s) from an input image
def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    face_cascade = cv2.CascadeClassifier('./openCV/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./openCV/haarcascade_eye.xml')
    # Load the image from a file or base64 encoded string
    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # Initialize an empty list to store the cropped faces
    cropped_faces = []
    for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2:
                cropped_faces.append(roi_color)
 #return cropped face
    return cropped_faces

                        
                     
                        
def get_b64_test_image_for_virat():
    # Read the base64 encoded string from a file
    with open("server/base64.txt") as f:
        return f.read()


if __name__ == '__main__':
    # Load the saved artifacts (model and class dictionaries)
    load_saved_artifacts()

    # Test the classify_image function with a base64 encoded string
    print(classify_image(get_b64_test_image_for_virat(), None))

'''
    classify_image(img_data, path) req: 
    
'''

    # Test the classify_image function with image files
    # print(classify_image(None, "./test_images/federer1.jpg"))
    # print(classify_image(None, "./test_images/federer2.jpg"))
    # print(classify_image(None, "./test_images/virat1.jpg"))
    # print(classify_image(None, "./test_images/virat2.jpg"))
    # print(classify_image(None, "./test_images/virat3.jpg")) 
    # Inconsistent result could be due to https://github.com/scikit-learn/scikit-learn/issues/13211
    # print(classify_image(None, "./test_images/serena1.jpg"))
    # print(classify_image(None, "./test_images/serena2.jpg"))
    # print(classify_image(None, "./test_images/sharapova1.jpg"))