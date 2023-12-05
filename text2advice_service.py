#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import sys
import traceback
from llm_client import *

# settting of import messaging
sys.path.append("messaging")
from messaging import *


class Text2AdviceService:
    def __init__(self):
        default_prompt = "sample default_prompt"
        self.llm_client = ChatGPTClient(default_prompt)

    def main_loop(self):
        while True:
            try:
                self.unit_work()
            except Exception as e:
                print(f"An error occurred while unit work: {e}")
                traceback.print_exc()

            time.sleep(10)

    def _make_response_and_publish(self, original_record, advice_text):
        original_record.advice_text = advice_text
        rec = original_record
        Text2AdviceServiceResMessaging().connect_and_basic_publish_record(rec)

    def unit_work(self):
        print("Getting new req from queue")
        rec = Text2AdviceServiceReqMessaging().connect_and_basic_get_record()
        if rec is None:
            return

        advice_text = self.llm_client.ask(rec.in_text)

        self._make_response_and_publish(rec, advice_text)


# print(WhisperAndPyannote().analyze("../rec1.wav"))
Text2AdviceService().main_loop()
