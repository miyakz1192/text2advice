#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import sys
import traceback
import uuid

# settting of import messaging
sys.path.append("messaging")
sys.path.append("messaging/service")
from messaging import *
from llm_mediator import *

class Text2AdviceService(LLMMediatorBase):
    # サービスへの要求キューからリクエストをgetした際にそれが許容されるものかどうかを判定する
    # 許容される・・・True、されない・・・False
    def is_acceptable_request(self, rec):
        if len(rec.in_text) <= 50:
            print("INFO: in_text is too short. skip ask to llm")
            return False
        return True

    # サービスへの要求キューからメッセージを取り出し、返す
    def get_from_req_queue(self):
        return Text2AdviceServiceReqMessaging().connect_and_basic_get_record()

    # LLMから結果を受け取り、サービスの結果返却キューにメッセージをpublishする
    def publish_to_res_queue(self, rec):
        Text2AdviceServiceResMessaging().connect_and_basic_publish_record(rec)

    # LLMInstanceから結果が帰ってこなかったレコードを、サービスの要求キューに入れなおす
    def publish_to_req_queue(self, rec):
        Text2AdviceServiceReqMessaging().connect_and_basic_publish_record(rec)

    # デフォルトの指示テキストを設定する
    def set_default_instruction(self):
        self.default_instruction = "以下の会話を見て、子の自己肯定感を高めるため、親はどのように振る舞うの一番良かったか、子育ての専門家の観点から、親にアドバイスしてください。また、入力部は親子の会話です。追加の指示として、子の自己肯定感を上げるために会話の内容をどのように修正すればよいか、修正案を提示してください。アドバイスと追加の指示は分けて提示してください。"

    # LLMInstanceServiceから処理が帰ってきた後、復帰値として、
    # サービス返却キューに返すレコードを作る
    # 引数にサービス要求キューに来たoriginal_recordが渡されるので、それに結果を代入して返す
    # record種別によってメンバーの名前が変わるため、個別実装になる。
    def _make_response_record(self, original_record, llm_output_text):
        original_record.advice_text = llm_output_text
        rec = original_record
        return rec

    # LLMInstanceに渡す入力テキスト(input_example)を生成する
    # たいていの場合、サービスへの要求キューに来たrecordが
    # ネタになるため、それを入力として受け取り処理する
    def _make_llm_input_text(self, rec):
        return rec.in_text #要求レコード内のin_text(会話の文字起こし結果)

Text2AdviceService().main_loop()
