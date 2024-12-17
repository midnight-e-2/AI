from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# generate speech by cloning a voice using default settings
tts.tts_to_file(text="안녕하세요 반갑습니다. 저는 티라노사우루스입니다!",
                file_path="data/dino_voice/xtts_result.wav",
                speaker_wav="data/target_voice/루피.wav",
                language="ko")
