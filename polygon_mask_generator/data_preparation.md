---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: Corrosion
    language: python
    name: python3
---

# <center>DATA preparation for EM image segmentation</center>


## 0. lib


```python
import os
import zipfile
import shutil
from PIL import Image
from read_roi import read_roi_file
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import numpy as np
from roifile import roiwrite, roiread
import re
from scipy.ndimage import zoom


```

## 1. crop + unzip + rename



Folder organization:
1. Within the folder named after the sheet, there should be three subfolders:
   1. ROI
      1. Side_1
      2. Side_2
   2. Images
      1. Side_1
      2. Side_2
   3. Masks
      1. Side_1
      2. Side_2
2. The ROI folder should end with "...11_measurements.zip", where 11 represents the sample number.
3. Image files should end with "...11.tif", where 11 represents the sample number.
4. When dealing with filenames not containing "Side_1" or "Side_2", these should be updated to:
   1. Rename files
   2. Definition of paths and constants


```python
# definition of paths and constants
#num=2 # this is number of 'side' can be edited based on name of images
specific = "010"
sampleType = "name" # excel sheet and type
typeOfMask = "masks"

#this number represents which layer of oxidation we need
numStartRoi = 0 #from 11
numEndRoi = 10 # till 20 


roiFolderSpecific=f'new_DATA/{sampleType}/roi/{specific}'
imagesFolderSpecific=f'new_DATA/{sampleType}/images/{specific}.png'


#the images and roi are sorted into folders 'roi' images'
roiFolder=f'new_DATA/{sampleType}/roi/'
masksFolder=f'new_DATA/{sampleType}/{typeOfMask}/'
imagesFolder=f'new_DATA/{sampleType}/images/'
excelFolder = f'new_DATA/name.xlsx'

#This needs to be changed based on psecific excel sheet and starting row based on "strana 1/2"

"""if num == 1:
    rowNum=0
else :
    rowNum= 150"""
rowNum=0


```

```python
def unzip_Remove (dir):
    for filename in os.listdir(dir):
        if filename.endswith(".zip"):
            folderName=os.path.splitext(filename)[0]
            folderPath= os.path.join(dir,folderName)
            os.makedirs(folderPath, exist_ok=True)
            with zipfile.ZipFile(os.path.join(dir, filename), 'r') as zipRef:
                zipRef.extractall(folderPath)
            os.remove(os.path.join(dir, filename))

unzip_Remove(roiFolder)
```

```python
def convertPng(tif_folder):
    tif_files = [f for f in os.listdir(tif_folder) if f.endswith('.tif')]
    for tifFile in tif_files:
        tifPath = os.path.join(tif_folder, tifFile)

        pngFileName = os.path.splitext(tifFile)[0] + '.png'
        pngPath = os.path.join(tif_folder, pngFileName)

        img = Image.open(tifPath)
        img.save(pngPath, 'PNG')

        os.remove(tifPath)

convertPng(imagesFolder)
```

```python
#sometimes the roi measurments had y2>y1 - was not consistent
def checkY2SmallerY1(roi_folder):
    for roiFolderName in os.listdir(roi_folder):
        roiFolderPath = os.path.join(roi_folder, roiFolderName)
        if os.path.isdir(roiFolderPath):
            for roi_file_name in os.listdir(roiFolderPath):
                if roi_file_name.endswith('.roi'):
                    roi_file_path = os.path.join(roiFolderPath, roi_file_name)
                    roi = roiread(roi_file_path)  
                    y1Value = roi.y1
                    y2Value = roi.y2
                    if y1Value < y2Value:
                        # Swap y1 and y2 values
                        roi.y1, roi.y2 = y2Value, y1Value
                    roiwrite(roi_file_path, roi)

checkY2SmallerY1(roiFolder)
```

```python
#this function is for renaming files so it is easier to read them, it is good to save them in folder that will have the original name

def renameFiles(folder, endIWthName,extension):
    for filename in os.listdir(folder):
        if filename.endswith(extension): 
            folderPath = os.path.join(folder, filename)
            #match = re.search(fr'strana{num}_(\d+){endIWthName}', filename)
            #THIS needs to be edited if different names of files
            match = re.search(fr'name_(\d+)', filename)
            if match:
                number = match.group(1).zfill(2)  # Format with leading zeros so no 2 but 02
                newFilename = f"{number}{extension}"
                newPath = os.path.join(folder, newFilename)
                
                # Rename the file
                os.rename(folderPath, newPath)
renameFiles(imagesFolder,"\.png",".png")
renameFiles(roiFolder,"_measurements","")
```

## 2. creation of MASKS

```python
#Check that the rois are reasonable
def showMeasurements(roiFolder, imgPath):
    roiFiles = [f for f in os.listdir(roiFolder) if f.endswith('.roi')]
    img = plt.imread(imgPath)
    plt.imshow(img)
    for roiFile in roiFiles:
        # Extract the number from the file name
        number = int(roiFile.split('.')[0])
        if number <= numEndRoi+10 and number > numStartRoi+10:
            roi_path = os.path.join(roiFolder, roiFile)
            roi = read_roi_file(roi_path)
        
            for key, value in roi.items():
                xValues = value['x1'], value['x2']
                yValues = value['y1'], value['y2']

                plt.plot(xValues, yValues, 'r')
    plt.show()
    
showMeasurements(roiFolderSpecific, imagesFolderSpecific)
```

### Creating mask

```python
def extractXY (img, numberOfSample, roi_folder, excelFilePath):
    excelFile = pd.read_excel(excelFilePath, sheet_name=f"{sampleType}",index_col=1)
    columns = [2] #columns where data from outer and complete oxidation are stored
    x1Values, x2Values, y1Values, y2Values = [None] * 10, [None] * 10, [None] * 10, [None] * 10 # create array, we have 10 measurments per one oxide layer

    plt.imshow(img)
    numberOfRoiLine, indexOrArr, nextLine = 1, -1, 0
    
    for col in columns:
        for row in range((numberOfSample - 1) * 10 + 1 + rowNum, (numberOfSample * 10) + 1+rowNum):
            callValue = excelFile.iloc[row - 1, col - 1]
            
            if  numberOfRoiLine > numStartRoi and numberOfRoiLine <= numEndRoi :#if it is the oxdiation we need
                indexOrArr += 1
                if callValue != 0: # if the measurment roi is valid
                    roi_path = os.path.join(roi_folder, f"{numberOfRoiLine}.roi")
                    roi = read_roi_file(roi_path)

                    if x1Values[indexOrArr] is None:
                        for key, value in roi.items():
                            x1Values[indexOrArr], x2Values[indexOrArr], y1Values[indexOrArr], y2Values[indexOrArr] = value['x1'], value['x2'], value['y1'], value['y2']
                    elif y2Values[indexOrArr] < value['y2']: # this part is for situations where we want to take more oxidation and take the biggest
                        for key, value in roi.items():
                            y2Values[indexOrArr] = value['y2']

                    if x1Values[indexOrArr] is not None and y1Values[indexOrArr] < value['y1']:
                        for key, value in roi.items():
                            y2Values[indexOrArr] = value['y2']
                


            # this part is for situations where we want to take more oxidations and take the biggest
            if numberOfRoiLine > (nextLine + 9):
                nextLine = nextLine + 10
                indexOrArr = -1
            numberOfRoiLine += 1
    print (x1Values)
    return x1Values, x2Values, y1Values, y2Values

```

```python
def createLine(x1Values,y1Values,x2Values,y2Values):
    xLine, y1Line, y2Line = [], [], []
    # Cubic spline interpolation on the filtered data points 
    if len(x1Values) > 3 and len(y1Values) > 3:
        splineInterpModelUpper = UnivariateSpline(x1Values, y1Values, s=0)
        xInterpValuesUpper = np.linspace(min(x1Values), max(x1Values), 1000)
        yInterpValuesUpper = splineInterpModelUpper(xInterpValuesUpper)
        # Clamp yInterpValuesUpper to be within the range of y1Values
        yInterpValuesUpper = np.clip(yInterpValuesUpper, min(y1Values), max(y1Values))
        xLine.extend(xInterpValuesUpper)
        y1Line.extend(yInterpValuesUpper)

    if len(x2Values) > 3 and len(y2Values) > 3:
        splineInterpModelLower = UnivariateSpline(x2Values, y2Values, s=0)
        xInterpValuesLower = np.linspace(min(x2Values), max(x2Values), 1000)
        yInterpValuesLower = splineInterpModelLower(xInterpValuesLower)
        yInterpValuesLower = np.clip(yInterpValuesLower, min(y2Values), max(y2Values))
        y2Line.extend(yInterpValuesLower)

    return xLine, y1Line, y2Line;
```

```python

def createPlotWithMask(img, numberOfSample, roiFolder, excelFilePath):
    if numberOfSample!=000:
        numberOfSample = (int)(numberOfSample/15)
    x1Values, x2Values, y1Values, y2Values = extractXY(img, numberOfSample, roiFolder, excelFilePath)

    print(f"Sample number: {numberOfSample}")
    
    # To remove None so it is continuous
    x1Values = [x for x in x1Values if x is not None]
    y1Values = [y for y in y1Values if y is not None]
    x2Values = [x for x in x2Values if x is not None]
    y2Values = [y for y in y2Values if y is not None]

    #save values of our cubic two lines
    xLine, y1Line, y2Line = createLine(x1Values,y1Values,x2Values,y2Values)

    plt.fill_between(xLine, y1Line, y2=y2Line, color='red', alpha=1)
    plt.plot(xLine, y1Line, 'red', marker='o', markersize=0.025, linestyle='-')
    plt.plot(xLine, y2Line, 'red', marker='o', markersize=0.025, linestyle='-')
    
    plt.axis('off')
    imageDimensions = img.shape[:2]
    originalWidth, originalHeight = imageDimensions[1], imageDimensions[0]
    
    plt.savefig(f'imageWithPlot_{numberOfSample}.png', bbox_inches='tight', pad_inches=0, dpi=originalWidth)
    plt.show()


    imgMask = cv2.imread(f'imageWithPlot_{numberOfSample}.png')
    hsv = cv2.cvtColor(imgMask, cv2.COLOR_BGR2HSV)
    lowerRed, upperRed = np.array([0, 100, 100]), np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lowerRed, upperRed)
    # Save the mask with the same width and length as the original image
    resized_mask = zoom(mask, (originalHeight / mask.shape[0], originalWidth / mask.shape[1]), order=0)
    cv2.imwrite(f"{masksFolder}/{'0' if numberOfSample < 10 else ''}{numberOfSample*15:03d}.png", resized_mask)
    os.remove(f'imageWithPlot_{numberOfSample}.png')

    
def createMasks(imagePath, roiFolder, excelFolder):
    pngImages = [f for f in os.listdir(imagePath) if f.endswith('.png')]  

    for pngImage in pngImages:
        # Extracting the index from the file name
        indexStr = os.path.splitext(pngImage)[0]
        try:
            subsNumber = int(indexStr)
        except ValueError:
            print(f"Failed to extract index from file name: {pngImage}")
            continue
        
        roiFolderMatch = os.path.join(roiFolder, indexStr)
        if not os.path.isdir(roiFolderMatch):
            print(f"No ROI folder found for subsNumber {subsNumber}")
            continue
        
        image = cv2.imread(os.path.join(imagePath, pngImage))
        if image is not None:
            createPlotWithMask(image, subsNumber, roiFolderMatch, excelFolder)
        else:
            print(f"Failed to read image: {pngImage}")

# Call the function with appropriate paths
createMasks(imagesFolder, roiFolder, excelFolder)

```

### Crop sides


```python
def crop_image_and_save(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read image: {img_path}")
        return

    cropped_img = crop_image(img)
    cv2.imwrite(img_path, cropped_img)

def crop_image(img):
    if len(img.shape) == 3:  
        height, width, _ = img.shape
    else: 
        height, width = img.shape

    #croping form left and right, the pic is split with 10 lines and left and right is half from one part
    #so it is 1/2, 1,1....,1,1,1/2 of parts
    left = int(width / 10 / 2)
    top = 0
    right = width - int(width / 10 / 2 )

    cropped_img = img[top:height, left:right]

    #uncomment if you want to see the pics
    """ 
    plt.imshow(cropped_img)
    plt.axis('off')
    plt.show()
    """

    return cropped_img

#crop masks
for filename in os.listdir(masksFolder):
    if filename.endswith('.png'): 
        img_path = os.path.join(masksFolder, filename)
        crop_image_and_save(img_path)

#crop images
for filename in os.listdir(imagesFolder):
    if filename.endswith('.png'):  
        img_path = os.path.join(imagesFolder, filename)
        crop_image_and_save(img_path)
```
