pip install moviepy openai-whisper pydub deep-translator gTTS

Make sure ffmpeg is installed and available in your system's PATH, as moviepy and pydub depend on it.
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

Audio Extraction: The video audio is extracted using moviepy.
Transcription: whisper is used to transcribe audio with timestamps.
Translation: deep-translator translates the transcription.
Text-to-Speech: gTTS synthesizes each translated segment into audio.
Combining Audio: The translated audio segments are combined and aligned with their original timestamps.
Audio Replacement: The translated audio is merged with the original video using moviepy.

