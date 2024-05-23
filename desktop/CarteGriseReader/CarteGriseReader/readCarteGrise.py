import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
import albumentations as albu
import os
from imutils.perspective import four_point_transform
import matplotlib.pyplot as plt
import easyocr
import json
import argparse
from flask import Flask, request, jsonify

app = Flask(__name__)

parser = argparse.ArgumentParser(description='Read Carte Grise')
parser.add_argument("--path", type=str, default='')
args = parser.parse_args()

model_path = 'pthF/unet_resnet34.onnx'

readerAr = easyocr.Reader(['ar'])
readerEng = easyocr.Reader(['en'])

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to coordinates.json
coordinates_path = os.path.join(current_dir, 'coordinates.json')

# Open coordinates.json
with open(coordinates_path, 'r') as myfile:
    data = myfile.read()
    coordinates = json.loads(data)

# Function to add padding to the image for better segmentation
def addPadding(padd, image):
    height, width = image.shape[:2]
    new_height = height + 2 * padd
    new_width = width + 2 * padd

    # Create a new image with padding
    padded_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # Copy the original image into the padded area
    padded_image[padd:padd + height, padd:padd + width] = image
    return padded_image

# Function to transform the image to the right format for the model
def load_transformed_image(image_path, margin):
    image = cv2.imread(image_path)
    image0 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image1 = addPadding(margin, image0)

    original_height, original_width, _ = image1.shape
    height, width, _ = image1.shape

    new_height = int(2 ** (np.ceil(np.log2(height / 12)) - 1) * 4)
    new_width = int(2 ** (np.ceil(np.log2(width / 12)) - 1) * 4)
    new_height = max(new_height, 640)
    new_width = max(new_width, 640)

    image = cv2.resize(image1, (new_width, new_height))

    transform = albu.Compose([albu.Normalize(p=1)], p=1)
    transformed = transform(image=image)
    return transformed['image'].transpose(2, 0, 1).astype(np.float32), image0, (original_height, original_width), image1

# Segmentation prediction function
def prediction(image, model_path, margin):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    transformed_image, originalImg, (height, width), paddImg = load_transformed_image(image, margin)
    transformed_image = np.expand_dims(transformed_image, axis=0)

    prediction = session.run([output_name], {input_name: transformed_image})[0]
    mask = prediction[0]

    mask = 1 / (1 + np.exp(-mask))
    mask = (mask > 0.5).astype(np.uint8)
    mask = mask.squeeze()
    mask = mask * 255
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return originalImg, mask, paddImg

# Preprocessing the mask image
def preprocess_mask(mask):
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    horizontal_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    vertical_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    combined_lines = cv2.bitwise_or(horizontal_lines, vertical_lines)
    return combined_lines

# Get the 4 coordinates for the corners of the image
def getCorners(segmentedImg):
    contours, _ = cv2.findContours(np.uint8(segmentedImg * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            corners = approx.reshape(-1, 2)
            return corners

# Crop the image based on the returned coordinates of the corners
def cropImg(path, model_path, margin, closeOperation=None):
    original_img0, output, paddedImg = prediction(path, model_path, margin)
    _, otsu_frame = cv2.threshold(output, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_frame = preprocess_mask(otsu_frame)
    kernel = np.ones((5, 5), np.uint8)
    otsu_frame = cv2.erode(otsu_frame, kernel, iterations=2)
    if closeOperation == True:
        otsu_frame = cv2.dilate(otsu_frame, kernel, iterations=5)
        otsu_frame = cv2.erode(otsu_frame, kernel, iterations=5)
    corners = getCorners(otsu_frame)
    result = four_point_transform(paddedImg, corners.reshape(4, 2))
    return result

# Crop only part of the image (Num d'immatriculation)
def returnPartOfImg(image, coordinates, target, carte):
    h, w, c = image.shape
    top_percent, bottom_percent, left_percent, right_percent = coordinates[carte][target]
    top = int(h * top_percent)
    bottom = int(h * bottom_percent)
    left = int(w * left_percent)
    right = int(w * right_percent)
    coord = image[top:bottom, left:right]
    return coord

# read the text inside the cropped image
def readPartOfImage(coord, readerEng, readerAr=None):
    resultEng = readerEng.readtext(coord)
    if readerAr is not None:
        resultAr = readerAr.readtext(coord)
        return resultEng, resultAr
    return resultEng

# read all the information in the face of "carte grise"
def getInformationAvant(result, coordinates, readerEng, readerAr):
    matr = returnPartOfImg(result, coordinates, "Immatriculation", "Avant")
    immAnt, numImm = readPartOfImage(matr, readerEng, readerAr)

    useDate = returnPartOfImg(result, coordinates, "UseDate", "Avant")
    try:
        l = readPartOfImage(useDate, readerEng)
        l1 = [i[1] for i in l]
        if len(l1) == 2:
            premMc, mcMar = l1
            mutation = "--"
        elif len(l1) > 2:
            premMc, mcMar, mutation = l1[:3]
        else:
            premMc, mcMar, mutation = "---"
    except:
        premMc, mcMar, mutation = "---"

    coord = returnPartOfImg(result, coordinates, "Usage", "Avant")
    try:
        usage = readPartOfImage(coord, readerEng)
        us = usage[-1][1]
        if us == "Usage":
            us = "--"
    except:
        us = "--"

    nom = returnPartOfImg(result, coordinates, "Nom", "Avant")
    try:
        name = readPartOfImage(nom, readerEng)
        n = [i[1] for i in name]
        prenom, nom = n[:2]
    except:
        prenom, nom = "--", "--"

    ad = returnPartOfImg(result, coordinates, "Adresse", "Avant")
    try:
        adresse = readPartOfImage(ad, readerEng)
        adr = [i[1] for i in adresse[1:]]
        adr = " ".join(adr)
    except:
        adr = "---"

    fin = returnPartOfImg(result, coordinates, "Validite", "Avant")
    try:
        dateValidite = readPartOfImage(fin, readerEng)
        validite = dateValidite[-1][1]
    except:
        validite = "--"

    return {"Carte Grise Avant": {
        "Numero d'immatriculation": numImm[0][1],
        "Immatriculation antérieure": immAnt[-1][1],
        "Premier M.C": premMc,
        "M.C au Maroc": mcMar,
        "Mutation le": mutation,
        "Usage": us,
        "Propriétaire": prenom + " " + nom,
        "Adresse": adr,
        "Fin de validité": validite
    }}

# read all the information in the back of "carte grise"
def getInformationArriere(result, coordinates, readerEng, readerAr):
    marq = returnPartOfImg(result, coordinates, "Marque", "Arriere")
    try:
        marque = readPartOfImage(marq, readerEng)
        m = [i[1] for i in marque]
        if len(m) >= 3:
            marque, typeV, genre = m[:3]
        else:
            marque, typeV, genre = "---"
    except:
        marque, typeV, genre = "---"

    mod = returnPartOfImg(result, coordinates, "Modele", "Arriere")
    try:
        model = readPartOfImage(mod, readerEng)
        mo = [i[1] for i in model]
        if len(mo) >= 2:
            model, typeCar = mo[-2:]
        elif len(mo) == 1:
            model, typeCar = "--", mo[0]
        else:
            model, typeCar = "--", "--"
        if model.isnumeric():
            model = "--"
        if typeCar.isnumeric():
            typeCar = "--"
    except:
        model, typeCar = "--", "--"

    numCh = returnPartOfImg(result, coordinates, "NumChassis", "Arriere")
    try:
        numChassis = readPartOfImage(numCh, readerEng)
        nc = numChassis[-1][1]
        if nc.isnumeric():
            nc = "--"
    except:
        nc = "---"

    nbrC = returnPartOfImg(result, coordinates, "NbrCylindres", "Arriere")
    try:
        nbCy = readPartOfImage(nbrC, readerEng)
        nbrCylindre = nbCy[0][1]
        if not nbrCylindre.isnumeric():
            nbrCylindre = "--"
    except:
        nbrCylindre = "--"

    puis = returnPartOfImg(result, coordinates, "PFiscale", "Arriere")
    try:
        pfis = readPartOfImage(puis, readerEng)
        pFiscales = pfis[0][1]
        if not pFiscales.isnumeric():
            pFiscales = "--"
    except:
        pFiscales = "--"

    nbPl = returnPartOfImg(result, coordinates, "NbrPlaces", "Arriere")
    try:
        nbrPl = readPartOfImage(nbPl, readerEng)
        nbrPlaces = nbrPl[0][1]
        if not nbrPlaces.isnumeric():
            nbrPlaces = "--"
    except:
        nbrPlaces = "--"
    try:
        poids = returnPartOfImg(result, coordinates, "PTAC", "Arriere")
        ptac = readPartOfImage(poids, readerEng)[0][1]
        if not ptac.lower().endswith('kg'):
            ptac = "--"
    except:
        ptac = "--"

    try:
        pds = returnPartOfImg(result, coordinates, "PoidsVide", "Arriere")
        pVide = readPartOfImage(pds, readerEng)[0][1]
        if not pVide.lower().endswith('kg'):
            pVide = "--"
    except:
        pVide = "--"

    try:
        pt = returnPartOfImg(result, coordinates, "PTMCT", "Arriere")
        ptmct = readPartOfImage(pt, readerEng)[0][1]
        if not ptmct.lower().endswith('kg'):
            ptmct = "--"
    except:
        ptmct = "--"

    return {"Carte Grise Arriere": {
        "Marque": marque,
        "Type": typeV,
        "Genre": genre,
        "Modèle": model,
        "Type Carburant": typeCar,
        "Numero du chassis": nc,
        "Nombre de cylindres": nbrCylindre,
        "Puissance fiscale": pFiscales,
        "Nombre de places": nbrPlaces,
        "P.T.A.C": ptac,
        "Poids à vide": pVide,
        "P.T.M.C.T": ptmct
    }}

def getResult(path, model_path):
    try:
        result = cropImg(path, model_path, 0)
    except:
        try:
            result = cropImg(path, model_path, 500, True)
        except:
            try:
                result = cropImg(path, model_path, 750, True)
            except:
                try:
                    result = cropImg(path, model_path, 1550, True)
                except:
                    result = "No corners detected"
    return result

def getInformation(path, model_path):
    result = getResult(path, model_path)
    if isinstance(result, np.ndarray):
        if result.shape[0] * result.shape[1] < 10000:
            return "There is a problem in this image, try another one with better quality"
        idAV = returnPartOfImg(result, coordinates, "id", "Avant")
        try:
            idA = readPartOfImage(idAV, readerEng)[-1][1].upper()
        except:
            idA = "--"
        if idA.endswith("GRISE") or idA.startswith("CARTE"):
            info = getInformationAvant(result, coordinates, readerEng, readerAr)
        else:
            idARR = returnPartOfImg(result, coordinates, "id", "Arriere")
            try:
                idAR = readPartOfImage(idARR, readerEng)[0][1].upper()
            except:
                idAR = "--"
            if idAR.startswith("MAR") or idAR.endswith("QUE"):
                info = getInformationArriere(result, coordinates, readerEng, readerAr)
            else:
                info = "Image wasn't read"
        return info
    else:
        return result

@app.route('/api/get_information', methods=['POST'])
def get_info():
    data = request.get_json()
    if 'path' in data:
        path = data['path']
        info = getInformation(path, model_path)
        return jsonify(info)
    else:
        return jsonify({"error": "Please provide 'path' parameter."})

if __name__ == '__main__':
    app.run(debug=True)
    
