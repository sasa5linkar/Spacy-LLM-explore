import os

arhiva2020 = "arhiva2020"
arhiva2022 = "arhiva2022"
jerteh_ner = "sr_jerteh_ner"

#folders and file paths

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESUORCES_DIR = os.path.join(ROOT_DIR, 'resources') 
COPRUS_DIR = os.path.join(ROOT_DIR, 'corpus')
SPACY_PIPELINE_DIR = os.path.join(ROOT_DIR, 'spacy_pipeline')
jerteh_ner_path = os.path.join(SPACY_PIPELINE_DIR, jerteh_ner)

model_id = "jerteh/gpt2-orao"
api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"

