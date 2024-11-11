import os
import torch
from datetime import datetime
import json
import tabulate
import textwrap


from langchain_openai import ChatOpenAI
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.messages import HumanMessage, SystemMessage

from transformers import pipeline

# import google.generativeai as genai
# import json

class LLMs:
    def __init__(self, load_default_api_model=False, load_default_open_source_model=False):
        self.openai_api_key = os.environ["OPENAI_API_KEY"]
        self.device = 0 if torch.cuda.is_available() else -1
        self.models = {
            "API_model" : None,
            "open_model" : None
            } ## Dictionary to store the models
        if load_default_api_model:
            self.load_api_model()
        if load_default_open_source_model:
            self.load_open_source_model()

    def load_api_model(self, model_name='gpt-3.5-turbo'):
        self.models["API_model"] = ChatOpenAI(model=model_name, api_key=self.openai_api_key)

    def load_open_source_model(self,
                               model_name='meta-llama/Llama-3.2-3B',
                               task='text-generation',
                               max_tokens=100):
        self.models["open_model"] = HuggingFacePipeline.from_model_id(model_id=model_name, 
                                                                      task=task,
                                                                      device=self.device,
                                                                      pipeline_kwargs={'max_new_tokens': max_tokens})

    def generate_text(self, model, prompt = ''):
        if model == 'API_model':
            return self.models[model].invoke(prompt).content
        else:
            return self.models[model].invoke(prompt)[len(prompt):].strip()

def test_llms(llms):

    prompts =[
        "Task: Can you tell my name given the text(Yes or No)? \n Text: I am from India. \n Output: ",
        "Task: Tell me should I get married (Yes or No) based on the text? \n Text: I am in love with someone who I think is fit for me, my parents agree with me and think that I should go ahead. I was 11 last year I am eager to get married. \n Output: ",
        "Task: Tell me the generation the person who spoke the text belongs to. \n Text: The burger place near my house is absolutely bussin'. \n Output: ",
        "Task: Identify the brand of the product mentioned in the text. \n Text: Yesterday I went to the store and bought this new phone, it is amazing and has this Siri feature which is really cool. \n Output: ",
        "Task: How many living beings are mentioned in the text? \n Text: Adam was going to this grandparents house on his horse, but stopped as he meet his old friend Rohan on the way. \n Output: ",
    ]
    expected_outputs = [
        "No",
        "No",
        "Gen Z",
        "Apple",
        "5"
    ]

    api_model_outputs = [llms.generate_text('API_model', prompt) for prompt in prompts]
    open_model_outputs = [llms.generate_text('open_model', prompt) for prompt in prompts]

    prompts = [prompt.replace("\n","") for prompt in prompts]
    api_model_outputs = [output.replace("\n","") for output in api_model_outputs]
    open_model_outputs = [output.replace("\n","") for output in open_model_outputs]

    def wrap_text(text, width=60):
        return "\n".join(textwrap.wrap(text, width))

    table_data = [
        [i + 1, wrap_text(prompts[i]), expected_outputs[i], wrap_text(api_model_outputs[i]), wrap_text(open_model_outputs[i])]
        for i in range(len(prompts))
    ]

    headers = ["#", "Prompt", "Expected Output", "API Model Output", "Open Model Output"]
    print(tabulate.tabulate(table_data, headers=headers, tablefmt="grid"))


class task3E:
    def __init__(self, api_key_gemini, api_key_openai):
        # self.api_key = api_key
        # self.device = device
        # self.models = {
        #     "API_model" : None,
        #     "open_model" : None
        #     } ## Dictionary to store the models
        self.gemini_api_key = api_key_gemini
        self.openai_api_key = api_key_openai
        self.model_gemini = None
        self.model_openai = None
        self.load_gemini_model()
        self.load_openai_model()


    def load_gemini_model(self, model_name='gemini-pro'):
        genai.configure(api_key=self.gemini_api_key)
        self.model_gemini = genai.GenerativeModel(model_name)
    
    def load_openai_model(self, model_name='gpt-3.5-turbo'):
        self.model_openai = ChatOpenAI(model=model_name, api_key=self.openai_api_key)


    def get_gemini_output(self, system_prompt, other_perspective, initial=False):
        if initial:
            response = self.model_gemini.generate_content(system_prompt)
            return response.text
        else:
            # Send a prompt to the model
            response = self.model_gemini.generate_content(system_prompt + "\n\n" + "Other Perspective: " + other_perspective)
        # print(response.text)
        return response.text
    
    def get_openai_output(self, system_prompt, other_perspective, initial=False):
        if initial:
            response = self.model_openai.invoke(system_prompt)
            return response.content
        else:
            response = self.model_openai.invoke(system_prompt + "\n\n" + "Other Perspective: " + other_perspective)
        return response.content
    
    def have_a_debate(self, system_task, system_prompt_initial_gemini, system_prompt_initial_openai, output_path=None):
        gemini_output = self.get_gemini_output(system_task + "\n\n" + system_prompt_initial_gemini, "", initial=True)
        print("Gemini Output: ", gemini_output)
        data_dict = {"model": "Gemini", 'prompt': system_task + "\n\n" + system_prompt_initial_gemini, 'response': gemini_output}
        data = [data_dict]

        openai_output = self.get_openai_output(system_task + "\n\n" + system_prompt_initial_openai, "", initial=False)
        print("OpenAI Output: ", openai_output)
        data_dict = {"model": "OpenAI", 'prompt': system_task + "\n\n" + system_prompt_initial_openai, 'response': openai_output}
        data.append(data_dict)

        if output_path != None:
            with open(output_path, "w") as f:
                f.write("Gemini Output: \n" + gemini_output + "\n\n")
                f.write("Openai Output: \n" + openai_output + "\n\n")

        for i in range(3):
            gemini_output = self.get_gemini_output(system_task, openai_output)
            data_dict = {"model": "Gemini", 'prompt': system_task, 'response': gemini_output}
            data.append(data_dict)
            print("OpenAI Output: \n", openai_output)
            print("\n\n")
            openai_output = self.get_openai_output(system_task, gemini_output)
            data_dict = {"model": "OpenAI", 'prompt': system_task, 'response': openai_output}
            data.append(data_dict)
            print("Gemini Output: \n", gemini_output)
            print("\n\n")
            if output_path != None:
                with open(output_path, "a") as f:
                    f.write("Gemini Output: \n" + gemini_output + "\n\n" + "Openai Output: \n" + openai_output + "\n\n")
        with open('./homework5/data.json', 'w') as f:
            json.dump(data, f, indent=4)
        
def read_prompts_from_file(filename):
    with open(filename, "r") as f:
        prompts = f.read().split("\n\n")
    return prompts


if __name__ == "__main__":
    llms = LLMs(load_default_api_model=True, load_default_open_source_model=True)

    # ## Task 1: Test the LLMs
    # test_llms(llms)

    # ## Task 3a: Jailbreaking the Models
    ## For writing in the text file. Seperate each prompt with a double newline.
    one_shot_prompts = read_prompts_from_file("one_shot_prompts.txt")
    multi_shot_prompts = read_prompts_from_file("multi_shot_prompts.txt")
    chain_of_thought_prompts = read_prompts_from_file("chain_of_thought_prompts.txt")

    print("Number of One Shot Prompts: ", len(one_shot_prompts))
    print("Number of Multi Shot Prompts: ", len(multi_shot_prompts))
    print("Number of Chain-of-Thought Prompts: ", len(chain_of_thought_prompts))

    ## initialize the data dictionary to store the data.
    data_dict = {
        "OneShot" : [],
        "MultiShot" : [],
        "ChainOfThought" : []
    }
    prompt_techniques = ["OneShot", "MultiShot", "ChainOfThought"]
    prompts = [one_shot_prompts, multi_shot_prompts, chain_of_thought_prompts]

    for prompt_technique, prompt_list in zip(prompt_techniques, prompts):
        for prompt in prompt_list:
            data_dict[prompt_technique].append(
                {
                    "Prompt": prompt, 
                    "API model response": llms.generate_text('API_model', prompt), 
                    "API model success": "1/0",
                    "Open model response": llms.generate_text('open_model', prompt),
                    "Open model success": "1/0"
                }
            )
    
    ## Saving the file.
    filename = f"data_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    with open(filename, 'w') as f:
        json.dump(data_dict, f, indent=4)
