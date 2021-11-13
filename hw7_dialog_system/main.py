import os
import re
import json

from asrecognition import ASREngine

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

import deeppavlov

from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters
)

import argparse


class AutomaticSpeechRecognition:
    def __init__(self):
        self.asr = ASREngine("ru", model_path="jonatasgrosman/wav2vec2-large-xlsr-53-russian")

    def __call__(self, path):
        res = self.asr.transcribe([path])
        return res[0]["transcription"]


class HiBuyIntentClassifier:
    def __init__(self):
        self.texts_hi = self.prepare_texts("data/hi.txt")
        self.texts_buy = self.prepare_texts("data/buy.txt")

    def clear_text(self, text):
        text = text.lower()
        text = re.sub("[^а-я]+", "", text)
        text = text.strip()
        return text

    def prepare_texts(self, path):
        with open(path) as file:
            texts = file.readlines()

        return [self.clear_text(text) for text in texts]

    def check_texts(self, texts, q_text):
        for text in texts:
            if q_text == text:
                return True

        return False

    def __call__(self, text):
        text = self.clear_text(text)

        if self.check_texts(self.texts_hi, text):
            return "intent_2_hi"

        if self.check_texts(self.texts_buy, text):
            return "intent_3_buy"

        return None


class Ru2EnTranslator:
    def __init__(self):
        self.model = model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
        self.tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
        self.tokenizer.src_lang = "ru"

    def __call__(self, text):
        encoded_ru = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.model.generate(**encoded_ru, forced_bos_token_id=self.tokenizer.get_lang_id("en"))
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]


class DomainIntentClassifier:
    def __init__(self):
        self.translator = Ru2EnTranslator()
        self.model = deeppavlov.build_model(deeppavlov.configs.classifiers.intents_snips, download=True)

    def __call__(self, text):
        text = self.translator(text)
        return self.model([text])[0]


class NaturalLanguageUnderstanding:
    def __init__(self):
        self.hi_buy_intent_classifier = HiBuyIntentClassifier()
        self.domain_intent_classifier = DomainIntentClassifier()

    def __call__(self, text):
        if len(re.sub("[^а-я]+", "", text.lower()).strip()) < 4:
            return "intent_1_not_understand"

        intent = self.hi_buy_intent_classifier(text)
        if intent is not None:
            return intent

        intent = self.domain_intent_classifier(text)
        if intent == "GetWeather":
            return "intent_4_get_weather"
        elif intent == "BookRestaurant":
            return "intent_5_book_restaurant"
        elif intent == "PlayMusic":
            return "intent_6_play_music"
        elif intent == "AddToPlaylist":
            return "intent_7_add_to_playlist"
        elif intent == "RateBook":
            return "intent_8_rate_book"
        elif intent == "SearchScreeningEvent":
            return "intent_9_search_screening_event"
        elif intent == "SearchCreativeWork":
            return "intent_10_search_creative_work"


class DialogManager:
    def __init__(self):
        with open("data/dm.json") as json_file:
            self.dict = json.load(json_file)

    def __call__(self, intent):
        return self.dict[intent]


class NaturalLanguageGeneration:
    def __init__(self, gen_text=True):
        self.gen_text = gen_text
        path = "data/nlg_text.json" if self.gen_text else "data/nlg_voice.json"
        with open(path) as json_file:
            self.dict = json.load(json_file)

    def is_text(self):
        return self.gen_text

    def __call__(self, answer):
        return self.dict[answer]


class Sendler():
    def send(self, update, context, out, is_text=True):
        if is_text:
            context.bot.send_message(chat_id=update.effective_chat.id, text=out)
        else:
            with open(out, "rb") as file:
                context.bot.send_voice(chat_id=update.effective_chat.id, voice=file)


class StartHandler(Sendler):
    def __init__(self, dm, nlg):
        self.dm = dm
        self.nlg = nlg

    def __call__(self, update, context):
        answer = self.dm("intent_0_start")
        out = self.nlg(answer)
        self.send(update, context, out, is_text=self.nlg.is_text())


class TextHandler(Sendler):
    def __init__(self, nlu, dm, nlg):
        self.nlu = nlu
        self.dm = dm
        self.nlg = nlg

    def __call__(self, update, context):
        text = update.message.text
        intent = self.nlu(text)
        answer = self.dm(intent)
        out = self.nlg(answer)
        self.send(update, context, out, is_text=self.nlg.is_text())


class VoiceHandler(Sendler):
    def __init__(self, asr, nlu, dm, nlg):
        self.asr = asr
        self.nlu = nlu
        self.dm = dm
        self.nlg = nlg

    def make_tmp_folder(self):
        if not os.path.exists("tmp"):
            os.makedirs("tmp")

    def download_voice(self, update, context):
        file_id = update.message.voice.file_id
        path = f"./tmp/{file_id}.ogg"

        self.make_tmp_folder()

        file = context.bot.get_file(file_id=update.message.voice.file_id)
        file.download(path)

        return path

    def remove_voice(self, path):
        os.remove(path)

    def __call__(self, update, context):
        voice = self.download_voice(update, context)
        text = self.asr(voice)
        self.remove_voice(voice)

        intent = self.nlu(text)
        answer = self.dm(intent)
        out = self.nlg(answer)
        self.send(update, context, out, is_text=self.nlg.is_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Say hello')
    parser.add_argument("--text", action="store_true", default=False)
    args = parser.parse_args()

    asr = AutomaticSpeechRecognition()
    nlu = NaturalLanguageUnderstanding()
    dm = DialogManager()
    nlg = NaturalLanguageGeneration(gen_text=args.text)

    start_handler = StartHandler(dm, nlg)
    text_handler = TextHandler(nlu, dm, nlg)
    voice_handler = VoiceHandler(asr, nlu, dm, nlg)

    with open("data/token.json") as json_file:
            token = json.load(json_file)["token"]

    updater = Updater(token=token)

    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler("start", start_handler))
    dispatcher.add_handler(MessageHandler(Filters.text & (~Filters.command), text_handler))
    dispatcher.add_handler(MessageHandler(Filters.voice, voice_handler))

    print("Ready")

    updater.start_polling()