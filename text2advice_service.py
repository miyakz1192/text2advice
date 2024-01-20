#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import sys
import traceback
import uuid

# settting of import messaging
sys.path.append("messaging")
from messaging import *


class Text2AdviceService:
    def __init__(self):
        self.default_instruction = "以下の会話を見て、子の自己肯定感を高めるため、親はどのように振る舞うの一番良かったか、子育ての専門家の観点から、親にアドバイスしてください。また、入力部は親子の会話です。追加の指示として、子の自己肯定感を上げるために会話の内容をどのように修正すればよいか、修正案を提示してください。アドバイスと追加の指示は分けて提示してください。"
        self.retry_target = []

    def main_loop(self):
        while True:
            try:
                self.unit_work()
            except Exception as e:
                print(f"An error occurred while unit work: {e}")
                traceback.print_exc()

            time.sleep(10)

    def _make_response_and_publish(self, original_record, advice_text):
        print("INFO: _make_response_and_publish")
        original_record.advice_text = advice_text
        rec = original_record
        Text2AdviceServiceResMessaging().connect_and_basic_publish_record(rec)

    def retry(self):
        if len(self.retry_target) == 0:
            return

        print("INFO: retry start")
        for rec in self.retry_target:
            print(f"INFO: -> {rec.id}")
            Text2AdviceServiceReqMessaging().connect_and_basic_publish_record(rec)

        self.retry_target = []
        print("INFO: retry end")

    def unit_work(self):
        print("Getting new req from queue")
        rec = Text2AdviceServiceReqMessaging().connect_and_basic_get_record()
        if rec is None:
            self.retry()
            return

        if len(rec.in_text) <= 50:
            print("INFO: in_text is too short. skip ask to llm")
            return

        advice_text = self.ask_to_llm(rec.in_text)
        if advice_text is None:
            # append to retry_list
            self.retry_target.append(rec)
            return

        self._make_response_and_publish(rec, advice_text)
        print("INFO: unit work end")

    def ask_to_llm(self, in_text):
        timed_out_counter = 0
        print("INFO: ask to llm")
        rec = LLMInstanceRecord(str(uuid.uuid4()), self.default_instruction, in_text)
        LLMInstanceReqMessaging().connect_and_basic_publish_record(rec)
        while True:
            try:
                print("INFO: waiting for llm")
                # FIXME: 単純に無限ループしていては、llminstanceが異常発生して落ちた時にこちらが気づかない。
                recv = LLMInstanceResMessaging().connect_and_basic_get_record()
                if recv is None:
                    print("INFO: no message")
                    time.sleep(10)
                    timed_out_counter += 1
                    # 1時間経過しても結果が帰って来ない場合
                    if timed_out_counter > 360:
                        print("INFO: timed out")
                        return None

                    continue

                if recv.id == rec.id:
                    print(f"INFO: got a message {recv.result}")
                    return recv.result
                else:
                    print(f"INFO: got a message but not eq ident {recv.id} != {rec.id}")
                    LLMInstanceResMessaging().connect_and_basic_publish_record(recv)

                time.sleep(10)
            except Exception as e:
                print(f"An error occurred while unit work: {e}")
                traceback.print_exc()


# print(WhisperAndPyannote().analyze("../rec1.wav"))
Text2AdviceService().main_loop()
