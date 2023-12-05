import openai
import os
from datetime import datetime

openai.api_key = os.getenv("CHAT_GPT_API_KEY")


class ChatGPTClient():
    def __init__(self, default_prompt=None):
        self.model_name = "gpt-4"
        self.default_prompt = default_prompt

    def ask(self, question):
        print("DEBUG: chat gpt in ask {self.default_prompt}+{question}")
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


class DummpyClient:
    def __init__(self, default_prompt=None):
        self.default_prompt = default_prompt

    def ask(self, question):
        return "sample res"
