import os
from fastapi import FastAPI, UploadFile, File
from transformers import pipeline
import shutil

os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers_cache"

app = FastAPI()

MODEL = "openai/whisper-small"
DEVICE = "cpu"  
print(f"Loading model: {MODEL}...")
transcriber = pipeline("automatic-speech-recognition",
                       model=MODEL,
                       device=DEVICE,
                       chunk_length_s=30,
                       return_timestamps=False)
print("Model loaded successfully!")

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):

    try:

        temp_file_path = f"/tmp/{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = transcriber(temp_file_path)
        text = result["text"].strip()


        os.remove(temp_file_path)

        print(f"Transcription completed: {text[:100]}...")
        return {"transcription": text}
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"message": "Whisper API is running"}
