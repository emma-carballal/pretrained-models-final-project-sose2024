"""Argilla annotation framework to display predictions of a single model"""

import os
import argilla as rg
import json
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY")
HF_SPACE_ARGILLA_TOKEN = os.getenv("HF_SPACE_ARGILLA_TOKEN")

# Connect to Argilla client
client = rg.Argilla(
    api_url=API_URL,
    api_key=API_KEY,
    headers={"Authorization": f"Bearer {HF_SPACE_ARGILLA_TOKEN}"}
)

# Dataset settings
settings = rg.Settings(
    guidelines="Review the linebreak prediction made by the model.",
    fields=[
        rg.TextField(
            name="original_sentence",
            title="Original Sentence",
            use_markdown=False,
        ),
        rg.TextField(
            name="predicted_sentence",
            title="Predicted Subtitle",
            use_markdown=False,
        ),
    ],
    questions=[
        rg.LabelQuestion(
            name="model_evaluation",
            title="Is the linebreak prediction correct?",
            labels=["Correct", "Incorrect"],
        ),
        rg.TextQuestion(
            name="corrected_linebreak",
            title="If incorrect, provide the corrected linebreak",
            required=False,
        )
    ],
)

# Create Argilla dataset
dataset = rg.Dataset(
    name="subtitle_linebreak_annotation_single_model",
    workspace="argilla",
    settings=settings,
    client=client,
)

# Create records
def load_jsonl_file(jsonl_file):
    """Loads a JSONL file and returns a list of records for a single model."""
    records = []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading JSONL data"):
            data = json.loads(line.strip())
            original_sentence = data['original_sentence']
            model_predictions = data['model_predictions']
            
            # Extract the predicted sentence from a single model
            predicted_sentence = model_predictions.get("predicted_sentence", "N/A")
                
            # Create a record for Argilla
            record = rg.Record(
                fields={
                    "original_sentence": original_sentence,
                    "predicted_sentence": predicted_sentence,
                },
                suggestions=[
                    rg.Suggestion(
                        question_name="model_evaluation",
                        value="Correct",
                    ),
                    rg.Suggestion(
                        question_name="corrected_linebreak",
                        value=predicted_sentence,
                    ),                    
                ] 
            )
            records.append(record)
    
    return records

jsonl_file = "data/evaluation/es/predictions/predictions.jsonl"
records = load_jsonl_file(jsonl_file)

# Log records to Argilla dataset
dataset.create()
dataset.records.log(records)

print(f"Logged {len(records)} records to the dataset.")
