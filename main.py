from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import tensorflow as tf
from pydantic import field_validator

import numpy as np
from fastapi.responses import JSONResponse
import joblib
from typing import List
from pydantic import BaseModel,  ValidationError

# Load the trained model and preprocessing objects
model = tf.keras.models.load_model("student_model.h5", compile=False)
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI()

class EEGData(BaseModel):
    eeg_signal: List[float]  # List of floats (eeg_signal)
    metadata: List[int]  # Metadata: [Gender (0/1), Age, MMSE]
    
    
    @field_validator('eeg_signal')
    @classmethod
    def validate_eeg_signal_length(cls, value):
        if len(value) != 1280:
            raise ValueError('eeg_signal must have exactly 1280 values')
        return value


   
@app.post("/predict/")
async def predict(data: EEGData):
    try:
        # Debug: print the received data
        print("Received data:", data)

        # Prepare the EEG signal and metadata for model input
        eeg_signal = np.array(data.eeg_signal, dtype="float32").reshape(1, -1, 1)
        metadata = np.array(data.metadata, dtype="float32").reshape(1, -1)

        # Debug: print the processed shapes
        print("Processed eeg_signal shape:", eeg_signal.shape)
        print("Processed metadata shape:", metadata.shape)

        # Normalize the metadata using the scaler
        metadata = scaler.transform(metadata)  # Normalize metadata

        # Predict with the model
        predictions = model.predict([eeg_signal, metadata])
        
        # Get the predicted class (use argmax to get the class index)
        predicted_class = np.argmax(predictions, axis=1)
        
        # Inverse transform the class index to the original label
        class_label = label_encoder.inverse_transform(predicted_class)[0]

        return {"predicted_class": class_label, "probabilities": predictions.tolist()}

    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(status_code=400, detail=str(e))

app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc: ValidationError):
    return JSONResponse(
        status_code=400,  # Return 400 instead of 422
        content={"detail": exc.errors()},
    )