from ultralytics import YOLO
import numpy as np
from enum import Enum

class BuildingProperty(Enum):
    DOOR = 0
    FACADE = 1
    OTHER_STREET_FURNITURE = 2
    POLE = 3
    SIGNAGE = 4
    SKY = 5
    TRASH_BIN = 6
    VEGETATION = 7
    VEHICLE = 8
    WINDOW = 9

class FacadeSegmentator:

    def __init__(self, modelDir):

        self.model = YOLO(modelDir)
        #Get the class names
        self.classNames = self.model.names  

    def getClassName(self, buildingProperty):
        if isinstance(buildingProperty, BuildingProperty):
            return self.classNames[buildingProperty.value]
        raise ValueError("Invalid BuildingProperty enum.")

    def getClassId(self, buildingProperty):
        if isinstance(buildingProperty, BuildingProperty):
            return buildingProperty.value
        raise ValueError("Invalid BuildingProperty enum.")

    def segmentFacadesInImage(self, image, confidenceThreshold=0.6):
        '''
        Segments facades and their properties in an image.
        Only detections with confidence >= confidenceThreshold are considered.
        Returns a dictionary where keys are BuildingProperty enums, and values are tuples of (masks, segmented images).
        '''
        results = self.model(image, verbose=False, overlap_mask=True)
        classSegmentation = {prop: ([], []) for prop in BuildingProperty}

        if results[0].masks is not None:
            confidences = results[0].boxes.conf.numpy()  # Get confidence scores as a numpy array
            for mask, classId, conf in zip(results[0].masks.data, results[0].boxes.cls, confidences):
                if conf >= confidenceThreshold:
                    mask = np.array(mask)
                    buildingProperty = BuildingProperty(int(classId))

                    # Apply the mask to the image to extract segmented regions
                    segmentedImage = image.copy()
                    segmentedImage[~mask.astype(bool)] = 0  # Apply mask

                    # Append the mask and segmented image to the respective class
                    classSegmentation[buildingProperty][0].append(mask)
                    classSegmentation[buildingProperty][1].append(segmentedImage)

        return classSegmentation