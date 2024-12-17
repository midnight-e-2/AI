from openai import OpenAI
import speech_recognition as sr

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

def whisper_tts(text):
    # whisper tts
    response = client.audio.speech.create(
        model='tts-1',
        input=text,
        voice='nova',
        response_format='mp3',
        speed=1.0
    )
    response.stream_to_file('data/dino_voice/whisper_result.mp3')
    
    # 추가 소리 변형
    audio = AudioSegment.from_wav("data/dino_voice/whisper_result.mp3")
    octaves = 0.9
    new_sample_rate = int(audio.frame_rate * (2.0 ** octaves))
    audio_high_pitch = audio._spawn(audio.raw_data, overrides={'frame_rate': new_sample_rate})
    audio_high_pitch = audio_high_pitch.set_frame_rate(44100)
    audio_high_pitch.export("data/dino_voice/whisper_result.wav", format="wav")
    