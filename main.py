from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel

class Item(BaseModel):
    text: str

app = FastAPI()
classifier = pipeline("sentiment-analysis")
ner_classifier = pipeline("ner")

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/predict/")
def predict(item: Item):
    return classifier(item.text)[0]

@app.post("/ner/")
def recognize_entities(item: Item):
    return ner_classifier(item.text)
