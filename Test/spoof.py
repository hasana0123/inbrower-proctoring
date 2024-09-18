# Libraries Importation
from ultralytics import YOLO
import uvicorn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO


# app intialization
app = FastAPI()

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

# Define a function for prediction
def predImage(image):
    # Load a model
    model = YOLO('best.pt')  # load a custom model
    # Predict with the model
    results = model(image, verbose=False )  # predict on an image
    for result in results:
        probs = result.probs  # Probs object for classification outputs
        if probs.top1 == 1:
            return f"Predicted Label: {model.names[probs.top1]}"
        elif probs.top1 == 0:
            return f"Predicted Label: {model.names[probs.top1]}"


@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    print("Image Loaded Successfully!")
    image = read_imagefile(await file.read())
    prediction = predImage(image)
    return prediction

if __name__ == "__main__":
    uvicorn.run(app, debug=True)