PRETRAINED_MODELS = [
    "bigscience/bloom-560m",
    "bigscience/bloom-1b1",
    "bigscience/bloom-1b7",
    "bigscience/bloom-3b",
    "bigscience/bloom-7b1",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b",
    "EleutherAI/gpt-j-6B",
]
#replace with bigscience/bloom-7b1 for actual training
PRETRAINED_MODEL = PRETRAINED_MODELS[0]

TRAINING_DATASETS = [
    "OpenAssistant/oasst1",
    "databricks/databricks-dolly-15k"
]

DEFAULT_TRAINING_DATASET = TRAINING_DATASETS[0]

DEFAULT_SEED=42