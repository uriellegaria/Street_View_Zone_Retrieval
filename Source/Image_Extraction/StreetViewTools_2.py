import streetview
import time
import numpy as np
import math
import shutil
import os
import cv2
import matplotlib.pyplot as plt
import json
import requests
from .street_view_lib import download_panorama_v3

class CollectionGeometry:
    '''
    The street geometry will collect images from both sides of the street from a start point to an end point. 
    The point geometry will collect images 360° around a point. 
    
    You can then stitch the images for analysis if needed.
    '''
    STREET = 0
    POINT = 1
    STREET_CONTINUOUS = 2
    COMBINED_VIEWS = 3
    STREET_BEARING = 4

class GoogleStreetViewCollector:
    
    def __init__(self, apiKey):
        #Api key is merely used for metadata collection, which is free
        self.apiKey = apiKey
        self.imagesPoint = 8
        self.size = 640
        self.pitch = 20
        self.fov = 90

    def getPanoramaMetadata(self, panoramaId):

        url = f"https://maps.googleapis.com/maps/api/streetview/metadata?pano={panoramaId}&key={self.apiKey}"
        response = requests.get(url)
        data = response.json()  # Convert to dictionary

        if "status" not in data:
            data["status"] = "ZERO_RESULTS"  # If missing, assume it's not found

        return data 
    
    def setPitch(self, pitch):
        self.pitch = pitch

    def setFOV(self, fov):
        self.fov = fov
    
    def setSize(self, size):
        self.size = size
    
    def setImagesStreet(self, nImagesStreet):
        self.imagesStreet = nImagesStreet
    
    def setImagesPoint(self, nImagesPoint):
        self.imagesPoint = nImagesPoint
    
    def getLocationString(self, latitude, longitude):
        return str(latitude)+","+str(longitude)

    def obtainDirectionBetweenLocations(self, location1, location2):
        '''
        Locations are in the order (latitude, longitude)
        '''

        lat1 = math.radians(location1[0])
        lat2 = math.radians(location2[0])

        diffLong = math.radians(location2[1] - location1[1])

        x = math.sin(diffLong) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)* math.cos(lat2)*math.cos(diffLong))

        initialBearing = math.atan2(x, y)


        initialBearing = math.degrees(initialBearing)
        compassBearing = (initialBearing + 360) % 360

        return compassBearing
    
    def getPanorama(self, panoramaId, zoom=3):
        image = download_panorama_v3(panoramaId, zoom=zoom)
        panoImage = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        return panoImage

    def xyz2lonlat(self, xyz):
        norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
        xyz_norm = xyz / norm

        x = xyz_norm[..., 0:1]
        y = xyz_norm[..., 1:2]
        z = xyz_norm[..., 2:]

        lon = np.arctan2(x, z)
        lat = np.arcsin(y)

        return np.concatenate([lon, lat], axis=-1)

    def lonlat2XY(self, lonlat, shape):
        """
        Same logic as the GitHub lonlat2XY(), but we can handle an extended width.
        shape = (height, width).
        """
        height, width = shape[0], shape[1]
        
        # Standard equirectangular mapping
        X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (width - 1)
        # No mod here; rely on borderMode for wrap, or extended image if we've duplicated columns.
        X = np.mod(X, width)
        Y = (lonlat[..., 1:]  / (np.pi) + 0.5) * (height - 1)

        return np.concatenate([X, Y], axis=-1)

    def extractView(self, equirectangular, heading=0, pitch=0, fov=90, width=640, height=640):
        """
        Adaptation of GitHub's GetPerspective(), plus dynamic overlap.
        """

        # 1) Detect Overlap
        hSrc, wSrc, _ = equirectangular.shape

        # 3) Compute Intrinsics
        f = 0.5 * width / np.tan(0.5 * fov / 180.0 * np.pi)
        cx = (width  - 1) / 2.0
        cy = (height - 1) / 2.0
        K = np.array([
            [f,   0,   cx],
            [0,   f,   cy],
            [0,   0,   1 ],
        ], np.float32)
        K_inv = np.linalg.inv(K)

        # 4) Generate pixel grid
        xVals = np.arange(width)
        yVals = np.arange(height)
        xVals, yVals = np.meshgrid(xVals, yVals)
        zVals = np.ones_like(xVals)

        xyz = np.concatenate([xVals[..., None], yVals[..., None], zVals[..., None]], axis=-1)
        xyz = xyz @ K_inv.T

        # 5) Rodrigues rotation
        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        R1, _ = cv2.Rodrigues(y_axis * np.radians(heading))
        R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(pitch))
        R = R2 @ R1
        xyz = xyz @ R.T

        # 6) Convert to lonlat
        lonlat = self.xyz2lonlat(xyz)

        # 7) Convert lonlat to XY with extended width
        XY = self.lonlat2XY(lonlat, shape=(hSrc, wSrc)).astype(np.float32)

        # 8) Remap using extended panorama
        perspective = cv2.remap(
            equirectangular, 
            XY[..., 0],  # x-coords
            XY[..., 1],  # y-coords
            interpolation=cv2.INTER_LANCZOS4, 
            borderMode=cv2.BORDER_WRAP
        )

        return perspective

        
    def collectPictures(self, collectionGeometry, outputPath, *args):
        if(collectionGeometry == CollectionGeometry.POINT):
            location = args[0]

            #Obtenemos el panorama en el punto especificado
            panoramaId = streetview.search_panoramas(lat = location[0], lon = location[1])
            if(len(panoramaId) > 0):
                panoramaId = panoramaId[0].pano_id
                panoramaMetadata = self.getPanoramaMetadata(panoramaId)
                if "status" not in panoramaMetadata or panoramaMetadata["status"] != "OK":
                    return 
                    
                panorama =self.getPanorama(panoramaId)

                currentAngle = -180
                deltaHeading = 360.0/self.imagesPoint
                for i in range(0, self.imagesPoint):
                    #Get the cut image
                    cutImage = self.extractView(panorama, heading = currentAngle, pitch = self.pitch)
                    outputPathPoint = os.path.join(outputPath, "image_point", f"gsv_{i}.jpg")
                    cv2.imwrite(outputPathPoint, cutImage)
                    currentAngle = (currentAngle + deltaHeading)%360
                
                #Save the metadata
                metadataOutputPath = os.path.join(outputPath, "image_point", "metadata.json")
                with open(metadataOutputPath, "w") as jsonFile:
                    json.dump(panoramaMetadata, jsonFile, indent=4)
        

        if(collectionGeometry == CollectionGeometry.COMBINED_VIEWS):
            location = args[0]
            bearing = args[1]
            outputPathStreet = args[2]
            pointIndex = args[3]

            panoramaId = streetview.search_panoramas(lat = location[0], lon = location[1])
            if(len(panoramaId) > 0):
                panoramaId = panoramaId[0].pano_id
                panoramaMetadata = self.getPanoramaMetadata(panoramaId)

                if panoramaMetadata is None or "status" not in panoramaMetadata or panoramaMetadata["status"] != "OK":
                    print("Invalid panorama")
                    return 

                panorama =self.getPanorama(panoramaId)


                currentAngle = -180
                deltaHeading = 360.0/self.imagesPoint
                for i in range(0,self.imagesPoint):
                    cutImage = self.extractView(panorama, heading = currentAngle, pitch = self.pitch)
                    sourcePathPoint = os.path.join(outputPath, "image_point")
                    if(not os.path.exists(sourcePathPoint)):
                        os.makedirs(sourcePathPoint)
                    
                    outputPathPoint = os.path.join(sourcePathPoint, f"gsv_{i}.jpg")
                    cv2.imwrite(outputPathPoint, cutImage)
                    currentAngle = (currentAngle + deltaHeading)%360

                metadataOutputPath = os.path.join(outputPath, "image_point", "metadata.json")
                with open(metadataOutputPath, "w") as jsonFile:
                    json.dump(panoramaMetadata, jsonFile, indent = 4)           

                #We will extract the left image now
                #Assuming bearing is the angle from true north
                angleLeftView = (-90)%360
                angleRightView = (90)%360

                sourcePathLeft = os.path.join(outputPathStreet, "Image_street_left")
                if(not os.path.exists(sourcePathLeft)):
                    os.makedirs(sourcePathLeft)

                pathImageLeft = os.path.join(sourcePathLeft, f"gsv_{pointIndex}.jpg")
                cutImageLeft = self.extractView(panorama, heading = angleLeftView, pitch = self.pitch)
                cv2.imwrite(pathImageLeft, cutImageLeft)


                sourcePathRight = os.path.join(outputPathStreet, "Image_street_right")
                if(not os.path.exists(sourcePathRight)):
                    os.makedirs(sourcePathRight)
                
                pathImageRight = os.path.join(sourcePathRight, f"gsv_{pointIndex}.jpg")
                cutImageRight = self.extractView(panorama, heading = angleRightView, pitch = self.pitch)
                cv2.imwrite(pathImageRight, cutImageRight)

        elif collectionGeometry == CollectionGeometry.STREET_BEARING:
            locations = args[0]
            bearings = args[1]  

            for idx, (location, bearing) in enumerate(zip(locations, bearings)):
                pano_ids = streetview.search_panoramas(lat=location[0], lon=location[1])
                if not pano_ids:
                    continue
                
                panoramaId = pano_ids[0].pano_id
                metadata = self.getPanoramaMetadata(panoramaId)
                if metadata is None or metadata.get("status") != "OK":
                    continue

                panorama = self.getPanorama(panoramaId)

                # Left/right views based on bearing
                left_heading = (-90) % 360
                right_heading = (90) % 360

                # Output directories
                pathLeft = os.path.join(outputPath, "Image_street_left")
                pathRight = os.path.join(outputPath, "Image_street_right")
                os.makedirs(pathLeft, exist_ok=True)
                os.makedirs(pathRight, exist_ok=True)

                # Extract and save
                left_img = self.extractView(panorama, heading=left_heading, pitch=self.pitch)
                right_img = self.extractView(panorama, heading=right_heading, pitch=self.pitch)

                cv2.imwrite(os.path.join(pathLeft, f"gsv_{idx}.jpg"), left_img)
                cv2.imwrite(os.path.join(pathRight, f"gsv_{idx}.jpg"), right_img)



        
            






            