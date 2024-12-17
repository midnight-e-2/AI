from openai import OpenAI
import speech_recognition as sr
import tempfile
# from config.path import SR_TEMP_DIR
from pydub import AudioSegment

client = OpenAI()

def whisper_speech_recognition(file):
    with open(file, 'rb') as audio_file:
        result = client.audio.transcriptions.create(
            file = audio_file,
            model = 'whisper-1',
            language='ko',
            response_format='text',
            temperature=0.0
        )
    return result

def whisper_tts(text, dinosaur_name):
    # whisper tts
    voice = 'shimmer'
    octaves = 0.9
    speed= 1.0
    if dinosaur_name == "티라노사우르스":
        # 중
        voice = 'shimmer'
        octaves = 0.9
        speed=1.0
    elif dinosaur_name == "딜로포사우루스":
        # 높
        voice = 'nova'
        octaves = 1.0
        speed=1.0
    elif dinosaur_name == "트리케라톱스":
        # 낮
        voice = 'onyx'
        octaves = 0.8
        speed=1.5     
        
    response = client.audio.speech.create(
        model='tts-1',
        input=text,
        voice=voice,
        response_format='wav',
        speed=speed
    )
    response.stream_to_file('data/dino_voice/whisper_result.wav')
    
    # 추가 소리 변형
    audio = AudioSegment.from_wav("data/dino_voice/whisper_result.wav")
    new_sample_rate = int(audio.frame_rate * (2.0 ** octaves))
    audio_high_pitch = audio._spawn(audio.raw_data, overrides={'frame_rate': new_sample_rate})
    audio_high_pitch = audio_high_pitch.set_frame_rate(44100)
    audio_high_pitch.export("data/dino_voice/whisper_result.wav", format="wav")
    
if __name__ =="__main__":
    whisper_tts("안녕 나는 티라노사우르스야 나는 아주 귀여워", "티라노사우르스")