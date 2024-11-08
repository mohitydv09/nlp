import os
import torch
import tabulate
import textwrap

from langchain_openai import ChatOpenAI
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.messages import HumanMessage, SystemMessage

from transformers import pipeline


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
                               max_tokens=20):
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

if __name__ == "__main__":
    llms = LLMs(load_default_api_model=True, load_default_open_source_model=True)

    ## Task 1: Test the LLMs
    test_llms(llms)

