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
MODEL_DIR="saved_models"

training_params={
    "num_epochs":15,
    "start_learning_rate":1e-5,
    "end_learning_rate":1e-6,
    "beta1":0.9,
    "beta2":0.95,
    "weight_decay":0.1,
    "batch_size":64 #using smaller model and larger batch size
}

INTRO_BLURB = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
)
INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"

#(instruction,response) pairs will be repeated for the duration of the conversation
PROMPT_FORMAT = """{intro}

{context}

{instruction_key}
{instruction}

{response_key}
{response}

{end_key}""".format(
    intro=INTRO_BLURB,
    context="{context}",
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
    response="{response}",
    end_key=END_KEY,
)
PROMPT_FORMAT_BEFORE_RESPONSE = """{intro}

{context}

{instruction_key}
{instruction}

{response_key}""".format(
    intro=INTRO_BLURB,
    context="{context}",
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)