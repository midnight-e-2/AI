from fastapi import FastAPI, File, UploadFile, Response, Form
from fastapi.responses import StreamingResponse
from io import BytesIO
import mimetypes
from fastapi.responses import JSONResponse
from model.dinosaur_model import Dinosaur_Model
from tts.whisper import whisper_speech_recognition, whisper_tts
from pydantic import BaseModel
from collections import defaultdict
import uvicorn
import os
import uuid
from config.path import DINO_VOICE_DIR, USER_VOICE_DIR

# FORM -> JSON
app = FastAPI()

user_dinosaur_models = defaultdict(dict)

# 문자 채팅 정보
class Chat_text(BaseModel):
    userNo:str
    user_chat: str
    dinosaur_name: str
    
@app.post('/chat-text')
async def chat_text(chat_text:Chat_text):
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
    audio_data = BytesIO(b'data/dino_voice/whisper_result.mp3')
    boundary = "myboundary"
    response_body = (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"text\"\r\n\r\n"
        f"{text_result}\r\n"
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"audio\"; filename=\"whisper_result.mp3\"\r\n"
        f"Content-Type: audio/mpeg\r\n\r\n"
        + audio_data.getvalue().decode("latin1") +  # 바이너리 데이터를 텍스트로 변환하여 포함
        f"\r\n--{boundary}--\r\n"
    )
    return Response(content=response_body)

# 소리 채팅 정보
@app.post('/chat-voice')
async def chat_voice(userNo:str = Form(...), dinosaur_name: str = Form(...), user_voice: UploadFile = File(...)):

    # speech recognition
    user_directory = os.path.join(USER_VOICE_DIR, userNo)
    
    if not os.path.exists(user_directory):
        os.makedirs(user_directory)
        print(f'폴더 {user_directory} 생성')
        
    file_uuid = str(uuid.uuid4())
    file_path = os.path.join(USER_VOICE_DIR, f'{userNo}/{file_uuid}.mp3')
    
    with open(file_path, 'wb') as buffer:
        buffer.write(await user_voice.read())
        
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
    audio_data = BytesIO(b'data/dino_voice/whisper_result.mp3')
    boundary = "myboundary"
    response_body = (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"text\"\r\n\r\n"
        f"{text_result}\r\n"
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"audio\"; filename=\"whisper_result.mp3\"\r\n"
        f"Content-Type: audio/mpeg\r\n\r\n"
        + audio_data.getvalue().decode("latin1") + 
        f"\r\n--{boundary}--\r\n"
    )
    return Response(content=response_body)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=7000)