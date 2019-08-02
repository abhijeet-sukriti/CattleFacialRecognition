from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from pathlib import Path
import uvicorn, aiohttp, asyncio
import base64, sys, numpy as np
import json
import io
from io import BytesIO
from PIL import Image


labels = ['Balwindr CL2','Gursewk CL11','Gursewk CL12','Gursewk CL3','Gursewk CL4','Gursewk CL5','Gursewk Shamdoo CL 2','Gursewk Shamdoo CL 3','Gursewk Shamdoo CL 4',
'Gursewk Shamdoo CL 8','Jaswnt CL 2','Jaswnt CL 3','Lovdp CL 1','mnpreet Cl 8','mnpreet cl 10','mohan cl 1','mohan cl 10','mohan cl 11','mohan cl 12','mohan cl 13',
'mohan cl 14','mohan cl 15','mohan cl 16','mohan cl 17','mohan cl 18','mohan cl 19','mohan cl 2','mohan cl 20','mohan cl 21','mohan cl 3','mohan cl 4','mohan cl 5',
'mohan cl 6','mohan cl 8','mohan cl 9','ramtej cl10','ramtej cl12','ramtej cl13','ramtej cl14','ramtej cl15','ramtej cl16','ramtej cl17','ramtej cl18','ramtej cl19',
'ramtej cl2','ramtej cl20','ramtej cl22','ramtej cl23','ramtej cl3','ramtej cl4','ramtej cl5','ramtej cl6','ramtej cl7','ramtej cl8','ramtej cl9']

path = Path(__file__).parent
#print (path)
#print (type(path))
model_file_url = 'https://www.dropbox.com/s/dk1z9ym2s19cpur/last4_vgg_face_cattle.h5?dl=1'
model_file_name = 'last4_vgg_face_cattle'

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

MODEL_PATH = path/'models'/f'{model_file_name}.h5'
IMG_FILE_SRC = path/'static'/'saved_image1.png'

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_model():
    #UNCOMMENT HERE FOR CUSTOM TRAINED MODEL
    await download_file(model_file_url, MODEL_PATH)
    #print (MODEL_PATH)
    #print (type(MODEL_PATH))
    model = load_model(str(MODEL_PATH)) # Load your Custom trained model
    model._make_predict_function()
    #model = ResNet50(weights='imagenet') # COMMENT, IF you have Custom trained model
    return model

# Asynchronous Steps
loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_model())]
model = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

# @app.route("/upload", methods=["POST"])
# async def upload(request):
#     data = await request.form()
#     print (type(data))
#     print('lllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllll')
#     print (data["file"])
#     img_bytes = await (data["file"].read())
#     #print (img_bytes)
#     bytes = base64.b64decode(img_bytes)
#     with open(IMG_FILE_SRC, 'wb') as f: f.write(bytes)
#     return model_predict(IMG_FILE_SRC, model)
#     #return model_predict(bytes, model)

@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    #print (type(data))
    #print('lllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllll')
    #print (data["file"])
    img_bytes = await (data["file"].read())
    #print (img_bytes)
    # bytes = base64.b64decode(img_bytes)
    with open(IMG_FILE_SRC, 'wb') as f: f.write(img_bytes)
    return model_predict(IMG_FILE_SRC, model)
    #return model_predict(bytes, model)


def model_predict(img_path, model):
    result = []; img = image.load_img(img_path, target_size=(224, 224))
    #img = Image.open(io.BytesIO(bytes))
    #result = []; img = Image.open(img_path, (224, 224))
    x = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
    #predictions = decode_predictions(model.predict(x), top=3)[0] # Get Top-3 Accuracy
    predictions = model.predict(x)
    #print (predictions)
    y_classes = predictions.argmax(axis=-1)[-1]
    #print (y_classes)
    predicted_label = sorted(labels)[y_classes]
    
    #print (predicted_label)
    #for p in predictions: _,label,accuracy = p; result.append((label,accuracy))
    result_html1 = path/'static'/'result1.html'
    result_html2 = path/'static'/'result2.html'
    #result_html = str(result_html1.open().read() +str(result) + result_html2.open().read())
    result_html = str(result_html1.open().read() +str(predicted_label) + result_html2.open().read())

    return HTMLResponse(result_html)

@app.route("/")
def form(request):
    index_html = path/'static'/'index.html'
    return HTMLResponse(index_html.open().read())

if __name__ == "__main__":
    if "serve" in sys.argv: uvicorn.run(app, host="0.0.0.0", port=8080)
