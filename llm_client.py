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


class DummyClient:
    def __init__(self, default_prompt=None):
        self.default_prompt = default_prompt

    def ask(self, question):
        return "sample res"
