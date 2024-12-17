from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from model.dinosaur_model import Dinosaur_Model
from tts.whisper import whisper_speech_recognition, whisper_tts
from pydantic import BaseModel
from collections import defaultdict
import uvicorn
import os
import base64
import uuid
from config.path import DINO_VOICE_DIR, USER_VOICE_DIR

app = FastAPI()

user_dinosaur_models = defaultdict(dict)

# 문자 채팅 정보
class Chat_text(BaseModel):
    userNo:int
    user_chat: str
    dinosaur_name: str
    
@app.post('/chat-text')
def chat_text(chat_text:Chat_text):
    userNo, user_chat, dinosaur_name = chat_text.userNo, chat_text.user_chat, chat_text.dinosaur_name
    if userNo not in user_dinosaur_models:
        if dinosaur_name not in user_dinosaur_models[userNo]:
            model = Dinosaur_Model(dinosaur_name)
            user_dinosaur_models[userNo][dinosaur_name] = model
    else:
        if dinosaur_name not in user_dinosaur_models[userNo]:
            model = Dinosaur_Model(dinosaur_name)
            user_dinosaur_models[userNo][dinosaur_name] = model
            
    text_result = user_dinosaur_models[userNo][dinosaur_name].exec(user_chat)
    
    # tts
    whisper_tts(text_result)
    file_path = os.path.join(DINO_VOICE_DIR, 'whisper_result.mp3')
    with open(file_path, "rb") as audio_file:
        voice_result = base64.b64encode(audio_file.read()).decode('utf-8') 
    result = {"text_chat": text_result, "voice_chat": voice_result}
    return JSONResponse(content={"dino_chat":result})

# 소리 채팅 정보
class Chat_voice(BaseModel):
    userNo:int
    user_voice: str
    dinosaur_name: str
@app.post('/chat-voice')
def chat_voice(chat_voice:Chat_voice):
    userNo, user_voice, dinosaur_name = str(chat_voice.userNo), chat_voice.user_voice, chat_voice.dinosaur_name
    
    # speech recognition
    user_directory = os.path.join(USER_VOICE_DIR, userNo)
    
    if not os.path.exists(user_directory):
        os.makedirs(user_directory)
        print(f'폴더 {user_directory} 생성')
        
    file_uuid = str(uuid.uuid4())
    file_path = os.path.join(USER_VOICE_DIR, f'{userNo}/{file_uuid}.mp3')
    
    user_voice_file = base64.b64decode(user_voice)
    
    with open(file_path, 'wb') as buffer:
        buffer.write(user_voice_file)
        
    sr_text = whisper_speech_recognition(file_path)
    
    # llm load
    if userNo not in user_dinosaur_models:
        if dinosaur_name not in user_dinosaur_models[userNo]:
            model = Dinosaur_Model(dinosaur_name)
            user_dinosaur_models[userNo][dinosaur_name] = model
    else:
        if dinosaur_name not in user_dinosaur_models[userNo]:
            model = Dinosaur_Model(dinosaur_name)
            user_dinosaur_models[userNo][dinosaur_name] = model
            
    text_result = user_dinosaur_models[userNo][dinosaur_name].exec(sr_text)
    
    # tts
    whisper_tts(text_result)
    
    file_path = os.path.join(DINO_VOICE_DIR, 'whisper_result.mp3')
    
    with open(file_path, "rb") as audio_file:
        voice_result = base64.b64encode(audio_file.read()).decode('utf-8') 
        
        result = {"text_chat": text_result, "voice_chat": voice_result}
    
    return JSONResponse(content={"dino_chat":result})

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=7000)