import io
import numpy as np
from fastapi import FastAPI,File,UploadFile
from fastapi.responses import StreamingResponse

import  torch
from torchvision import models,transforms
from PIL import Image,ImageDraw

from torchvision.models.detection import fasterrcnn_resnet50_fpn

NUM_CLASSES = 2  # background + your classes (MUST match training)

# Step 1: recreate architecture
model = fasterrcnn_resnet50_fpn(num_classes=NUM_CLASSES)

# Step 2: load weights
state_dict = torch.load("artifacts\\models\\fasterrcnn.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()
device='cpu'
model.to(device)

transform=transforms.Compose([
    transforms.ToTensor(),
])

app=FastAPI()

def predict_and_draw(image:Image.Image):
    img_tensor=transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions=model(img_tensor)
        
    prediction=predictions[0]
    boxes=prediction['boxes'].cpu().numpy()
    labels=prediction['labels'].cpu().numpy()
    scores=prediction['scores'].cpu().numpy()
    
    img_rgb=image.convert('RGB')
    draw=ImageDraw.Draw(img_rgb)
    for box,score in zip(boxes,scores):
        if score>0.7:
            x_min,y_min,x_max,y_max=box
            draw.rectangle([x_min,y_min,x_max,y_max],outline='red',width=3)
    return img_rgb

@app.get('/')
def read_root():
    return {"message":'welcome to gun detection project'}

@app.post('/predict/')
async def predict(file:UploadFile=File(...)):
    image_data= await file.read()
    image= Image.open(io.BytesIO(image_data))
    output_image=predict_and_draw(image)
    image_byte_arr=io.BytesIO()
    output_image.save(image_byte_arr,format='PNG')
    image_byte_arr.seek(0)
    return StreamingResponse(image_byte_arr,media_type='image/png')


    