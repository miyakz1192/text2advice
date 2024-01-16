# import openai
import os
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# openai.api_key = os.getenv("CHAT_GPT_API_KEY")


class ChatGPTClient():
    def __init__(self, default_prompt=None):
        self.model_name = "gpt-4"
        self.default_prompt = default_prompt

    def ask(self, question):
        print(f"DEBUG: chat gpt in ask {self.default_prompt}+{question}")
        start_time = datetime.now()

        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "user",
                 "content": self.default_prompt + "\n" + question},
            ],
        )

        end_time = datetime.now()
        res = response.choices[0]["message"]["content"].strip()
        print(res)
        print(f"elapsed time: {end_time - start_time}")

        with open(
            f"output-{self.model_name}-{datetime.now().strftime('%Y%m%dT%H%M%S')}.txt", "w"
        ) as f:
            f.write(f"model: {self.model_name}\n")
            f.write("time: " + str(end_time - start_time) + "\n")
            f.write("question: " + question + "\n")
            f.write("answer: " + res + "\n")

        return res


class ElyzaClient:
    def __init__(self, default_prompt=None):
        # トークナイザーとモデルの準備
        self.tokenizer = AutoTokenizer.from_pretrained(
            "elyza/ELYZA-japanese-Llama-2-7b-instruct"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "elyza/ELYZA-japanese-Llama-2-7b-instruct",
            torch_dtype=torch.float32,
            device_map="auto",
            offload_folder="/home/miyakz/offload_folder"
        )
        self.default_prompt = default_prompt

    def ask(self, question):
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

        tokenizer = self.tokenizer
        model = self.model

        start_time = datetime.now()

        prompt = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
            bos_token=tokenizer.bos_token,
            b_inst=B_INST,
            system=f"{B_SYS}{self.default_prompt}{E_SYS}",
            prompt=self.default_prompt + "\n" + question,
            e_inst=E_INST,
        )

        with torch.no_grad():
            token_ids = tokenizer.encode(prompt, add_special_tokens=False,
                                         return_tensors="pt")
            output_ids = model.generate(
                token_ids.to(model.device),
                max_new_tokens=1024,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        res = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):],
                               skip_special_tokens=True)

        print(f"DEBUG: elyza in ask {self.default_prompt}+{question}")
        end_time = datetime.now()
        print(res)
        print(f"elapsed time: {end_time - start_time}")

        with open(
            f"output-ELYZA-{datetime.now().strftime('%Y%m%dT%H%M%S')}.txt", "w"
        ) as f:
            f.write(f"model: ELYZA\n")
            f.write("time: " + str(end_time - start_time) + "\n")
            f.write("question: " + question + "\n")
            f.write("answer: " + res + "\n")

        return res


class SwallowClient:

    def __init__(self, default_prompt=None):
        model_name = "tokyotech-llm/Swallow-7b-instruct-hf"
        tokenizer_model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                # torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                load_in_8bit=False,
                device_map="auto")

        self.default_prompt = default_prompt
        self.do_Lora = False

    def ask(self, question):
        if self.do_Lora is True:
            # ベースモデルのロードと同様にCPU環境だと、torch_dtype=torch.float32を指定
            self.model = PeftModel.from_pretrained(
                self.model,
                args.lora_data,
                # torch_dtype=torch.float16,
                torch_dtype=torch.float32,
                load_in_8bit=False,
            )
            # ベースモデルとLoRAの重みをマージしないと上手く動作しない。
            self.model = self.model.merge_and_unload()

        Do_sample = True  # @param {type:"boolean"
        temperature = 0.99  # @param {type:"slider", min:0, max:2, step:0.1}
        top_p = 0.95  # @param {type:"slider", min:0, max:1, step:0.01}
        #max_new_tokens = 128  # @param {type:"slider", min:128, max:1024, step:64}
        max_new_tokens = 1024  # @param {type:"slider", min:128, max:1024, step:64}

        instruction_example = self.default_prompt
        input_example = question
        prompt = self.create_prompt(instruction_example, input_example)

        input_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
            return_tensors="pt"
        )

        tokens = self.model.generate(
            input_ids=input_ids.to(device=self.model.device),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=Do_sample,
        )

        res = self.tokenizer.decode(tokens[0], skip_special_tokens=True)

        start_time = datetime.now()
        print(f"DEBUG: swallow in ask {self.default_prompt}+{question}")
        end_time = datetime.now()
        print(res)
        print(f"elapsed time: {end_time - start_time}")

        with open(
            f"output-swallow-{datetime.now().strftime('%Y%m%dT%H%M%S')}.txt", "w"
        ) as f:
            f.write(f"model: swallow\n")
            f.write("time: " + str(end_time - start_time) + "\n")
            f.write("question: " + question + "\n")
            f.write("answer: " + res + "\n")

        return res

    def create_prompt(self, instruction, input=None):
        PROMPT_DICT = {
            "prompt_input": (
                "以下に、あるタスクを説明する指示があり、それに付随する入力が更なる文脈を提供しています。"
                "リクエストを適切に完了するための回答を記述してください。\n\n"
                "### 指示:\n{instruction}\n\n### 入力:\n{input}\n\n### 応答:"
            ),
            "prompt_no_input": (
                "以下に、あるタスクを説明する指示があります。"
                "リクエストを適切に完了するための回答を記述してください。\n\n"
                "### 指示:\n{instruction}\n\n### 応答:"
            ),
        }
        if input:
            # Use the 'prompt_input' template when additional input is provided
            return PROMPT_DICT["prompt_input"].format(
                    instruction=instruction, input=input)
        else:
            # Use the 'prompt_no_input' template
            # when no additional input is provided
            return PROMPT_DICT["prompt_no_input"].format(instruction=instruction)


class DummyClient:
    def __init__(self, default_prompt=None):
        self.default_prompt = default_prompt

    def ask(self, question):
        return "sample res"


# This class must be defined the bottom of this file
def get_client_class():
    if "LLM_CLIENT_NAME" in os.environ:
        return globals()[os.environ["LLM_CLIENT_NAME"]]
    else:
        # return default client
        return ElyzaClient
