import os
from fastapi import FastAPI, UploadFile, File
from transformers import pipeline
import shutil

# Set cache directory for Transformers (optional for Railway compatibility)
os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers_cache"

# Initialize FastAPI app
app = FastAPI()

# Load Whisper model
MODEL = "openai/whisper-small"
DEVICE = "cpu"  # Use CPU for simplicity on Railway
print(f"Loading model: {MODEL}...")
transcriber = pipeline("automatic-speech-recognition",
                       model=MODEL,
                       device=DEVICE,
                       chunk_length_s=30,
                       return_timestamps=False)
print("Model loaded successfully!")

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Endpoint to transcribe audio files
    """
    try:
        # Save the uploaded file temporarily
        temp_file_path = f"/tmp/{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Transcribe the audio
        result = transcriber(temp_file_path)
        text = result["text"].strip()

        # Clean up the temporary file
        os.remove(temp_file_path)

        print(f"Transcription completed: {text[:100]}...")
        return {"transcription": text}
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        return {"error": str(e)}

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Whisper API is running"}
