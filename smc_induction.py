# import datasets

# # Load the dataset
# dataset = datasets.load_dataset("barc0/gpt-4_description_with_llama_codegen")

# # Print the dataset
# print(dataset)
# breakpoint()


from arc import train_problems, validation_problems
import os
import re
# from prompt import get_common_lib_from_file
import json
import numpy as np
import tiktoken
from datasets import Dataset
from tqdm import tqdm
import random
from arc.read import parse_dir
from arc.types import ArcIOPair, ArcProblem

VERSION = "v3"

# EXTRA_NEWLINE = "\n"
EXTRA_NEWLINE = "\n"
TRANSPOSE = False

# =================== inference params ===================
import time
from transformers import AutoTokenizer

TEMPERATURE = 1.0
BASE_MODEL = "barc0/Llama-3.1-ARC-Potpourri-Induction-8B"
LORA_DIR = None
# LORA_DIR = "barc0/barc-llama3.1-8b-instruct-lora64-induction-gpt4omini35k_lr2e-4_epoch3"

BATCH_SIZE = 64
num_of_samples_per_problem = 256
TENSOR_PARALLEL = 4
# ========================================================

COLOR_MAPPING = {
0: "Black",
1: "Blue",
2: "Red",
3: "Green",
4: "Yellow",
5: "Grey",  # instead of "Grey"
6: "Pink",
7: "Orange",
8: "Teal",
9: "Maroon"
}

COLOR_REPLACEMENTS = {
    "Grey": "Gray",
    "Teal": "Purple",
    "Maroon": "Brown",
}

# Fix Color Mapping
for k, v in COLOR_MAPPING.items():
    if v in COLOR_REPLACEMENTS:
        COLOR_MAPPING[k] = COLOR_REPLACEMENTS[v]

# Map a hard coded color to a deterministic some other color in source code, keeping cases same
def color_deterministic(problem_source_code, old_color, new_color):
    upper_template = f"(((?<=[^a-zA-Z])|^)({old_color.upper()})(?=[^a-zA-Z]|$))"
    capitalized_template = (
        f"(((?<=[^a-zA-Z])|^)({old_color.lower().capitalize()})(?=[^a-zA-Z]|$))"
    )
    lower_template = f"(((?<=[^a-zA-Z])|^)({old_color.lower()})(?=[^a-zA-Z]|$))"

    # Do findall operation with this regex
    upper_regex = re.compile(upper_template)
    capitalized_regex = re.compile(capitalized_template)
    lower_regex = re.compile(lower_template)

    replace_upper = re.sub(
        upper_regex, lambda x: new_color.upper(), problem_source_code
    )

    replace_capitalized = re.sub(
        capitalized_regex,
        lambda x: new_color.lower().capitalize(),
        replace_upper,
    )

    replace_lower = re.sub(
        lower_regex,
        lambda x: new_color.lower(),
        replace_capitalized,
    )

    return replace_lower


def test_color_deterministic():
    problem_source_code = "teal, Teal, TEAL"
    ret = color_deterministic(problem_source_code, "teal", "purple")
    print(ret)


def convert_color_name(text, mapping):
    for old_color, new_color in mapping.items():
        text = color_deterministic(text, old_color, new_color)
    return text

def test_convert_color_name():
    text = "teal, Teal, TEAL\nMaroon COLOR>MAROON, maroon"
    ret = convert_color_name(text, COLOR_REPLACEMENTS)
    print(ret)


class IOPair:
    x: np.ndarray
    y: np.ndarray
    def __init__(self, x, y):
        self.x = x
        self.y = y
        # check type
        assert isinstance(self.x, np.ndarray)
        assert isinstance(self.y, np.ndarray)
        # check shape
        assert len(self.x.shape) == 2
        assert len(self.y.shape) == 2

class Problem(ArcProblem):
    # typing hint for the members
    uid: str
    filename: str
    seed_id: str
    code: str
    train_pairs: list
    test_pairs: list

    def __init__(self, uid, filename=None, code=None, seed_id=None, train_pairs=None, test_pairs=None):
        self.uid = uid
        self.filename = filename
        self.seed_id = None
        if filename:
            self.seed_id = filename.split(".")[0]
            if "_" in self.seed_id:
                self.seed_id= self.seed_id.split("_")[0]
        if seed_id:
            self.seed_id = seed_id
        if self.seed_id:
            pattern = r"[0-9a-f]{8}"
            assert re.match(pattern, self.seed_id)
            self.load_arc_problem(self.seed_id)

        self.code = code
        if train_pairs:
            self.train_pairs = train_pairs
        if test_pairs:
            self.test_pairs = test_pairs

        # assert self.code, "Code is not provided"
        assert self.train_pairs, "Train pairs are not provided"
        assert self.test_pairs, "Test pairs are not provided"
        # check type
        assert isinstance(self.train_pairs, list)
        assert isinstance(self.test_pairs, list)
        for pair in self.train_pairs:
            assert isinstance(pair, ArcIOPair), "Train pair is not of type IOPair. it is of type {}".format(type(pair))
        for pair in self.test_pairs:
            assert isinstance(pair, ArcIOPair), "Test pair is not of type IOPair. it is of type {}".format(type(pair))
        assert all(isinstance(pair, ArcIOPair) for pair in self.train_pairs)
        assert all(isinstance(pair, ArcIOPair) for pair in self.test_pairs)


    def load_arc_problem(self, seed_id):
        # using train_problems
        arc_problem = None
        for problem in train_problems + validation_problems:
            if problem.uid == seed_id:
                arc_problem = problem
                break
        assert arc_problem is not None
        self.train_pairs = []
        for pair in arc_problem.train_pairs:
            self.train_pairs.append(IOPair(pair.x.T, pair.y.T))
        self.test_pairs = []
        for pair in arc_problem.test_pairs:
            self.test_pairs.append(IOPair(pair.x.T, pair.y.T))

def grid_to_input(grid, transpose: bool):
    if transpose:
        transformed_grid = grid.T
    else:
        transformed_grid = grid
    return "\n".join(" ".join(COLOR_MAPPING[c] for c in row) for row in transformed_grid) + EXTRA_NEWLINE

def make_problem_input_str(problem: Problem, transpose: bool):
    prompt ="Given input-output grid pairs as reference examples, carefully observe the patterns to predict the output grid for new test input. Each pair follows the same transformation rule. Grids are 2D arrays represented as strings, with cells (colors) separated by spaces and rows by newlines."
    prompt += "\nHere are the input and output grids for the reference examples:\n"
    for i, pair in enumerate(problem.train_pairs):
        prompt += f"Example {i+1}\n"
        prompt += f"Input:\n{grid_to_input(pair.x, transpose)}\nOutput:\n{grid_to_input(pair.y, transpose)}\n\n" 
    prompt += "Here is the input grid for the test example:\n"
    prompt += "Input:\n" + "\n".join(grid_to_input(pair.x, transpose) for pair in problem.test_pairs)
    return prompt

def make_input_prompt_induction(problem: Problem, transpose: bool):
    common_lib_prefix = ""
    question = common_lib_prefix + make_problem_input_str(problem, transpose=transpose)
    question += "\nWrite a Python function `transform` that can convert any given input grid to its corresponding output grid based on the pattern observed in the reference examples."
    return question

DEFAULT_SYSTEM_PROMPT_IND = "You are a world-class puzzle solver with exceptional pattern recognition skills and expertise in Python programming. Your task is to analyze puzzles and provide Python solutions."

def convert_chat_format_induction(question, answer):
    messages =  {
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT_IND},
            {"role": "user", "content": question},
        ]
    }
    if answer:
        messages["messages"].append({"role": "assistant", "content": answer})
    return messages

def get_concept_arc_problems():
    problems = []
    for problem_directory in os.listdir("/om/user/evan_kim/966/probARC/ConceptARCSmallSequential"):
        problems.extend(parse_dir("/om/user/evan_kim/966/probARC/ConceptARCSmallSequential/"+problem_directory))
    return problems

def main():
    # ====================================================================================
    # ==================== starting induction prompt input generation ====================
    # ====================================================================================

    import argparse
    SAVE_DIR = "/om/user/evan_kim/966/probARC/induction_inputs_outputs"

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_seeds", action="store_true")
    parser.add_argument("--use_concept_arc", action="store_true", default=True)
    parser.add_argument("--load_file", type=str)
    parser.add_argument("--load_huggingface_dataset", type=str)
    parser.add_argument("--output_huggingface_dataset", type=str, required=False, default=None)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

    train_data_induction = []
    # TODO: actually, for the seed_problems, should NOT transpose
    problems = []
    if args.use_concept_arc:
        concept_arc_problems = get_concept_arc_problems()
        uids = [p.uid for p in concept_arc_problems]
        assert len(uids) == len(set(uids))

        for problem in concept_arc_problems:
            for ti, test_pair in enumerate(problem.test_pairs):
                new_problem = Problem(uid=f"{problem.uid}-{ti}",
                                         train_pairs=problem.train_pairs,
                                         test_pairs=[test_pair])
                problems.append(new_problem)

    for problem in problems:
        question = make_input_prompt_induction(problem, transpose=TRANSPOSE)
        answer = f"""Let's solve this puzzle using Python code with the common library functions. We'll first reason about the problem and then write the code to solve it. The `transform` function will take the input grid and return the output grid. Here is the Python code with the comments describing how to solve the problem:
```python
{problem.code}
```
""" 
        answer = convert_color_name(answer, COLOR_REPLACEMENTS)
        messages = convert_chat_format_induction(question, answer)
        # Add uid to the data structure
        train_data_induction.append({"uid": problem.uid, "messages": messages["messages"]})

    print("==============input=============")
    print(train_data_induction[0]["messages"][1]["content"])
    print("==============output=============")
    print(train_data_induction[0]["messages"][2]["content"])

    token_counts_ind = []
    filtered_train_data_induction = []
    filtered_train_data_id = []
    for cnt, data_induction in enumerate(train_data_induction):
        token_count_ind = 0 
        token_count_ind += len(tokenizer.encode(data_induction["messages"][0]["content"]))
        token_count_ind += len(tokenizer.encode(data_induction["messages"][1]["content"]))
        token_count_ind += len(tokenizer.encode(data_induction["messages"][2]["content"]))

        if token_count_ind < 8000:
            filtered_train_data_induction.append(data_induction)
            token_counts_ind.append(token_count_ind)
            filtered_train_data_id.append(cnt)

    print('Induction')
    print(f"Total number of tokens: {sum(token_counts_ind)}")
    print(f"Averge number of tokens per example: {sum(token_counts_ind) / len(token_counts_ind)}")
    print(f"Max number of tokens per example: {max(token_counts_ind)}")
    print("Filtered indices:", len(filtered_train_data_id))

    # Save the filtered data
    split_filename = "concept_arc" if args.use_concept_arc else "seeds"
    if args.use_concept_arc and args.use_seeds:
        split_filename = "concept_arc_and_seeds"
    
    import datetime
    # Get current date and time
    datetime_str = datetime.datetime.now().strftime("%m%d%H%M%S")

    problem_file = f"arc_problems_{split_filename}_{len(filtered_train_data_induction)}_{datetime_str}.jsonl"
    if TRANSPOSE:
        problem_file = f"arc_problems_{split_filename}_{len(filtered_train_data_induction)}_transpose_{datetime_str}.jsonl"
    if EXTRA_NEWLINE:
        problem_file = f"arc_problems_{split_filename}_{len(filtered_train_data_induction)}_extra_newline_{datetime_str}.jsonl"
    if TRANSPOSE and EXTRA_NEWLINE:
        problem_file = f"arc_problems_{split_filename}_{len(filtered_train_data_induction)}_transpose_extra_newline_{datetime_str}.jsonl"

    if VERSION:
        problem_file = problem_file.replace(".jsonl", f"_{VERSION}.jsonl")
    
    target_problem_filepath = os.path.join(SAVE_DIR, problem_file)
    print(f"Saving to {target_problem_filepath}")
    with open(target_problem_filepath, "w") as f:
        f.write("\n".join(json.dumps(p) for p in filtered_train_data_induction))

    # Print summary statistics
    print('Induction')
    print(f"Total number of tokens: {sum(token_counts_ind)}")
    print(f"Average number of tokens per example: {sum(token_counts_ind) / len(token_counts_ind)}")
    print(f"Max number of tokens per example: {max(token_counts_ind)}")
    print(f"Total number of filtered examples: {len(filtered_train_data_id)}")

    # ====================================================================================
    # =========================== starting induction inference ===========================
    # ====================================================================================

    if LORA_DIR:
        tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)
    else:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # import json
    data = []
    # problem_file = "./arc_problems_train_327.jsonl"
    # problem_file = "./arc_problems_validation_400.jsonl"
    # problem_file = "./arc_problems_selected-val-subset50_50_extra_newline.jsonl"
    # problem_file = "./arc_problems_selected-train-subset50_50.jsonl"

    # problem_file = "./arc_problems_selected-train-subset50_50_extra_newline.jsonl"
    # problem_file = "./arc_problems_train_327_extra_newline.jsonl"
    # problem_file = "./arc_problems_validation_400_extra_newline.jsonl"
    # problem_file = "./arc_problems_validation_400_extra_newline_v2.jsonl"
    
    # problem_file = "/om/user/evan_kim/966/probARC/induction_inputs_outputs/arc_problems_concept_arc_128_transpose_extra_newline_v3.jsonl"
    problem_file = target_problem_filepath

    with open(problem_file) as f:
        for line in f:
            data.append(json.loads(line))

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    if LORA_DIR:
        llm = LLM(model=BASE_MODEL, enable_lora=True, max_lora_rank=256, max_model_len=12000,
                enable_prefix_caching=True, tensor_parallel_size=TENSOR_PARALLEL)
        lora_request=LoRARequest("barc_adapter", 1, LORA_DIR)
    else:
        llm = LLM(model=BASE_MODEL, enable_lora=False, max_model_len=12000,
                enable_prefix_caching=True, tensor_parallel_size=TENSOR_PARALLEL)

    import datetime
    datetime_str = datetime.datetime.now().strftime("%m%d%H%M%S%f")
    if LORA_DIR:
        saving_file = f"{problem_file.replace('.jsonl', '')}_{LORA_DIR.split('/')[-1]}_temp_{TEMPERATURE}_{datetime_str}.jsonl"
    else:
        saving_file = f"{problem_file.replace('.jsonl', '')}_{BASE_MODEL.split('/')[-1]}_temp_{TEMPERATURE}_{datetime_str}.jsonl"
    print(f"Saving to {saving_file}")
    time.sleep(5)

    from tqdm import tqdm
    all_responses = []
    for d in tqdm(data):
        messages = d["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        inputs = tokenizer.apply_chat_template([
            {"role":"system", "content":messages[0]["content"]},
            {"role":"user", "content":messages[1]["content"]}
        ], tokenize=False, add_generation_prompt=True)
        input_tokens = tokenizer.apply_chat_template([
            {"role":"system", "content":messages[0]["content"]},
            {"role":"user", "content":messages[1]["content"]}
        ], tokenize=True, add_generation_prompt=True)
        # print(inputs)
        print(f"Number of tokens: {len(input_tokens)}")
        if len(input_tokens) > 8000:
            print("skip!!!!!")
            continue

        assert num_of_samples_per_problem % BATCH_SIZE == 0
        if  len(input_tokens) < 1750:
            tmp_batch_size = BATCH_SIZE * 4
        elif len(input_tokens) < 4000:
            # double the number of samples
            tmp_batch_size = BATCH_SIZE * 4
        elif len(input_tokens) < 5000:
            tmp_batch_size = BATCH_SIZE 
        else:
            tmp_batch_size = BATCH_SIZE

        print(f"batch size: {tmp_batch_size}")
        sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=1536,
                                        n=tmp_batch_size)
        aggregate_outputs = []
        for i in range(num_of_samples_per_problem // tmp_batch_size):
            if LORA_DIR:
                outputs = llm.generate(
                    inputs,
                    sampling_params,
                    lora_request=lora_request
                )
            else:
                outputs = llm.generate(
                    inputs,
                    sampling_params,
                ) 
            aggregate_outputs.append(outputs)

        if not aggregate_outputs:
            breakpoint()
        else:
            # print(aggregate_outputs[0])
            print("\n================ ")


        # Print the outputs.
        responses = []
        for outputs in aggregate_outputs:
            for output in outputs:
                # prompt = output.prompt
                # print(f"Prompt: {prompt!r}")
                for i in range(len(output.outputs)):
                    generated_text = output.outputs[i].text
                    # print(f"Generated text: {generated_text!r}\n")
                    responses.append(generated_text)

        all_responses.append({"uid": d["uid"], "prompt":inputs , "responses": responses, "base_model": BASE_MODEL, "lora_dir": LORA_DIR})

        with open(saving_file, "w") as f:
            f.write("\n".join(json.dumps(p) for p in all_responses))

    print(f"Saving to {saving_file}")

    time.sleep(15)
    

if __name__ == "__main__":
    main()
