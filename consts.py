SUGGESTED_INPUT_MODELS = [
    "distilgpt2",
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
#TODO: replace with largest model that can fit on single GPU for actual training (bloom 7b1)
DEFAULT_INPUT_MODEL="bigscience/bloom-1b1"

PEFT=True

TRAINING_DATASETS = [
    "OpenAssistant/oasst1",
    "databricks/databricks-dolly-15k"
]

DEFAULT_TRAINING_DATASET = TRAINING_DATASETS[0]



TRAINING_PARAMS={
    "num_epochs":15,
    "test_size":100, #only use 100 test examples 
    "start_learning_rate":1e-5,
    "end_learning_rate":1e-6,#TODO: check how to set up linear decay to this number instead of default 0 in huggingface trainer
    "beta1":0.9,
    "beta2":0.95,
    "weight_decay":0.1,
    "batch_size":64,
    "model_copies_per_device":1,
}

DEFAULT_SEED=42
DEFAULT_MODEL_PATH=f"{DEFAULT_TRAINING_DATASET.rsplit('/',1)[-1]}-{DEFAULT_INPUT_MODEL.rsplit('/',1)[-1]}-batch_size_{TRAINING_PARAMS['batch_size']}-{TRAINING_PARAMS['num_epochs']}_epochs"

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
