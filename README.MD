python3.11 tested
pip install numpy==2.0
python3 -m pip install -U demucs
pip install moviepy pydub gTTS deep-translator
pip install coqui-tts
pip install colorlog

Make sure ffmpeg is installed and available in your system's PATH, as moviepy and pydub depend on it.
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

- Extract audio from source video
- Separate extracted audio to background sounds and voice
- Transcribe to text from separated voice with time segments
- Translate all segments by deep_translator by google
- Learn pre-trained TTS model by source voice
- Synthesize trained voice to translated text
- Mix background sounds with synthesized segments
- Replace translated audio with background to source video

