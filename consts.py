SUGGESTED_INPUT_MODELS = [
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
#replace with largest model that can fit on single GPU for actual training (probably bloom1b7)
DEFAULT_INPUT_MODEL="bigscience/bloom-560m"

TRAINING_DATASETS = [
    "OpenAssistant/oasst1",
    "databricks/databricks-dolly-15k"
]

DEFAULT_TRAINING_DATASET = TRAINING_DATASETS[0]

DEFAULT_SEED=42
DEFAULT_MODEL_PATH="oasst1-bloom-560m-finetuned"

TRAINING_PARAMS={
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
PREV_RESPONSE_KEY = "### Previous response:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
RESPONSE_KEY_NL = f"{RESPONSE_KEY}\n"

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