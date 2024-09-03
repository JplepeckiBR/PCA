from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
import io

app = FastAPI()

@app.get("/")
def root():
    return "Hello World"


@app.post("/flip-image")
async def flip_image(file: UploadFile = File(...)):
    image = Image.open(file.file)
    flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    img_io = io.BytesIO() # Create an in-memory bytes buffer.
    flipped_image.save(img_io, 'PNG') # Save the flipped image to the buffer in JPEG format.
    img_io.seek(0)# Move the file pointer to the beginning of the buffer.
    return StreamingResponse(img_io, media_type="image/png") #officially type for JPEG images

#Type Annotation (file: UploadFile):Tells FastAPI that this parameter should be treated as an uploaded file.
#Enables the use of UploadFile methods and attributes, such as file.file, file.filename, file.content_type, etc.
