import os
import shutil
import uuid
from typing import Union
from typing import Annotated
from fastapi import FastAPI, Body, File, UploadFile
from facedb import FaceDB
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

face_db = FaceDB(
    path="facedata",
    metric="euclidean",
    database_backend="chromadb",
    embedding_dim=128,
    module="face_recognition",
)

def upload_file_to_dir(file: UploadFile, todir: str):
    # create dir if not exists
    if not os.path.exists(todir):
        os.makedirs(todir)
    # generate unique filename
    filename = str(uuid.uuid4())
    file_ext = file.filename.split(".")[-1]
    fullname = filename + "." + file_ext
    filepath = f"{todir}/{fullname}"
    
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return fullname
        
@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.delete("/delete-face-all")
def delete_face_all():
    return face_db.delete_all()


@app.post("/add-face")
def add_face(name: Annotated[str, Body()], file: UploadFile = File(...)):
    dir = "stroage"
    # dir = "tmp"
    filename = upload_file_to_dir(file, dir)
    img_path = f"{dir}/{filename}"
    id_face = None
    try:
        id = face_db.add(name, img=img_path)
        # os.remove(img_path)
    except ValueError as e:
        print(e)
        return {"error": str(e)}
    except Exception as e:
        print(e)
        return {"error": str(e)}
    return face_db.get(id)

@app.get("/faces")
def get_faces():
    # select all faces
    return face_db.all()

@app.post("/find-face")
def find_face(file: UploadFile = File(...)):
    dir = "tmp"
    filename = upload_file_to_dir(file, dir)
    img_path = f"{dir}/{filename}"
    result = None
    result = face_db.recognize(img=img_path)
    os.remove(img_path)
    if not result:
        return []
        
    for r in result:
        r['data'] = face_db.get(r['id'])
    return result
    

if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)