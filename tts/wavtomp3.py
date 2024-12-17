from pydub import AudioSegment
import os

def wav2mp3(file_path, save_path):
    audio = AudioSegment.from_wav(file_path)
    file_name =  os.path.basename(file_path)
    mp3_file_path = os.path.join(save_path, f'{file_name}.mp3')
    audio.export(mp3_file_path, format('mp3'))
    return mp3_file_path
