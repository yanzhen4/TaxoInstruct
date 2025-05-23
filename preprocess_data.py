import json
from tqdm import tqdm
import fire

def preprocess_data(
    input_data_path: str = None,
    output_data_path: str = None,
    baselines: bool = False
    ):

    print("input_data_path: ", input_data_path)
    print("output_data_path: ", output_data_path)

    setexpan_input_prompt = "Find other entities belonging to the category {} and sharing the same granularity as the seeds {}"
    setexpan_output_prompt = "The expanded entities are {}"

    taxoexpan_input_prompt = "Given candidate parents {}, find the parent class for {}"
    taxoexpan_output_prompt = "The parent class is {}"

    find_parent_input_prompt = "Find the parent class for {}"
    find_parent_output_prompt = "The parent class is {}"

    print("Set-Expan input prompt: ", setexpan_input_prompt)
    print("Set-Expan output prompt: ", setexpan_output_prompt)
    print()

    print("Taxo-Expan input prompt: ", taxoexpan_input_prompt)
    print("Taxo-Expan output prompt: ", taxoexpan_output_prompt)
    print()

    print("Find-Parent input prompt: ", find_parent_input_prompt)
    print("Find-Parent output prompt: ", find_parent_output_prompt)
    print()

    def formatting_prompts_func(example):
        
        if example['task'] == 'Taxo-Expan':
            candidate_parents_str = "{" + ", ".join(example['candidate_parents']) + "}"
            input_prompt = taxoexpan_input_prompt.format(candidate_parents_str, example['child'])

            if 'parent' in example:
                parent = example['parent']
            else:
                parent = ""

            output_prompt = taxoexpan_output_prompt.format(parent)

            if baselines:
                return { "input" : input_prompt, "output" : output_prompt, 'task': example['task']}

        elif example['task'] == 'Set-Expan':
            input_str  = "{" + ", ".join(example['input']) + "}"

            if 'output' in example:
                output_str  = "{" + ", ".join(example['output']) + "}"
            else:
                output_str = ""

            input_prompt = setexpan_input_prompt.format(example['parent'], input_str)
            output_prompt = setexpan_output_prompt.format(output_str)
        
        elif example['task'] == 'Find-Parent':
            input_str = "{" + ", ".join(example['input']) + "}"
            input_prompt = find_parent_input_prompt.format(input_str)

            if 'parent' in example:
                parent = example['parent']
            else:
                parent = ""
            
            output_prompt = find_parent_output_prompt.format(parent)
        
        return { "input" : input_prompt, "output" : output_prompt, 'task': example['task']}

    with open(output_data_path, 'w') as fout:
        with open(input_data_path) as fin:
            for line in tqdm(fin):
                example = json.loads(line)
                data_dict = formatting_prompts_func(example)
                json.dump(data_dict, fout)
                fout.write('\n')

if __name__ == "__main__":
    fire.Fire(preprocess_data)