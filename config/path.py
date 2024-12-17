import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT_DIR, 'data')
DINO_VOICE_DIR = os.path.join(DATA_DIR, 'dino_voice')
TARGET_VOICE_DIR = os.path.join(DATA_DIR, 'target_voice')
USER_VOICE_DIR = os.path.join(DATA_DIR, 'user_voice')
