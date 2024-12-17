from pydub import AudioSegment

audio = AudioSegment.from_mp3('data/target_voice/루피.mp3')
audio.export('data/target_voice/루피.wav', format('wav'))
