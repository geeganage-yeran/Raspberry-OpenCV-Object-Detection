import cv2

# Threshold to detect objects

classNames = []  # List to store class names
classFile = "/home/pi/Desktop/Object_Detection_Files/coco.names"  # Path to file containing class names

# Read class names from the file
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/pi/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"  # Path to model configuration file
weightsPath = "/home/pi/Desktop/Object_Detection_Files/frozen_inference_graph.pb"  # Path to model weights file

# Load pre-trained model
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)  # Set input size for the model
net.setInputScale(1.0/ 127.5)  # Scale factor for the model
net.setInputMean((127.5, 127.5, 127.5))  # Mean subtraction for input image normalization
net.setInputSwapRB(True)  # Swap red and blue channels

def getObjects(img, thres, nms, draw=True, objects=[]):
    """
    Detect objects in the input image using the DNN model.

    Args:
        img: Input image to process.
        thres: Confidence threshold for object detection.
        nms: Non-maxima suppression threshold to reduce overlapping boxes.
        draw: Boolean to indicate whether to draw bounding boxes on the image.
        objects: List of target objects to detect (default: detect all objects).

    Returns:
        img: Processed image with bounding boxes (if draw=True).
        objectInfo: List of detected objects with their bounding boxes and class names.
    """
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)  # Perform object detection
    if len(objects) == 0:  # If no specific objects are specified, detect all objects
        objects = classNames

    objectInfo = []  # Initialize list to store object information
    if len(classIds) != 0:  # If objects are detected
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]  # Get the class name of the detected object
            if className in objects:  # Check if the detected object is in the target objects list
                objectInfo.append([box, className])  # Append object info to the list
                if draw:  # If draw=True, draw bounding boxes and labels on the image
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)  # Draw bounding box
                    cv2.putText(img, className.upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)  # Draw class name
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)  # Draw confidence percentage

    return img, objectInfo  # Return the processed image and detected objects information

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # Initialize video capture (webcam)
    cap.set(3, 640)  # Set frame width
    cap.set(4, 480)  # Set frame height

    while True:
        success, img = cap.read()  # Read a frame from the webcam
        result, objectInfo = getObjects(img, 0.45, 0.2)  # Perform object detection on the frame
        cv2.imshow("Output", img)  # Display the processed frame
        cv2.waitKey(1)  # Wait for a key press for 1 ms