from fastapi import FastAPI
import numpy as np
import uvicorn  
from io import BytesIO
from keras.models import load_model
from fastapi import FastAPI, File, UploadFile
from starlette.responses import RedirectResponse
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import uvicorn
import asyncio

# Define application
app = FastAPI(
    title="i-Eye Saffron",
    description="Predict whether a saffron product is real or fake",
    version="0.1",
)

model = load_model('vgg_model_2.hdf5')

def predict(image: Image.Image):
    image = np.asarray(image.resize((224, 224)))[..., :3]  
    image = np.expand_dims(image, 0)  
    image = image / 127.5 - 1.0    
    result = model.predict(image)    
    resp = {}  
    resp["class"] = "Saffron" if np.argmax(result)==1 else "Non-Saffron"  
    resp["Probability of Non-Saffron"] = f"{result[0][0]*100:0.2f} %"  
    resp["Probability of Saffron"] = f"{result[0][1]*100:0.2f} %"  
    return resp

def read_imagefile(file) -> Image.Image:  
    image = Image.open(BytesIO(file))  
    return image

@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):  
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")  
    if not extension:  
        return "Image must be jpg or png format!"  
    image = read_imagefile(await file.read())  
    prediction = predict(image)
    return prediction

if __name__ == "__main__":  
    uvicorn.run(app,debug=True)