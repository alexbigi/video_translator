import logging
import shutil
import subprocess

from moviepy.editor import VideoFileClip, AudioFileClip
import whisper
from gtts import gTTS
from pydub import AudioSegment
import os
from deep_translator import GoogleTranslator
import torch
from TTS.api import TTS
from colorlog import ColoredFormatter
from pydub.effects import speedup

LOG_LEVEL = logging.DEBUG
LOG_FORMAT = "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
logging.root.setLevel(LOG_LEVEL)
formatter = ColoredFormatter(LOG_FORMAT)
stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
stream.setFormatter(formatter)
log = logging.getLogger('pythonConfig')
log.setLevel(LOG_LEVEL)
log.addHandler(stream)


# Step 1: Extract audio from video
def extract_audio(video_path, audio_output_path):
    log.debug("EXTRACT AUDIO - PROCESS")
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_output_path)
    video.close()
    log.debug("EXTRACT AUDIO - COMPLETE")


def run_demucs(input_audio_path, output_directory='demucs_output'):
    # also can use Spleeter
    log.debug("SEPARATE AUDIO - PROCESS")
    command = [
        'demucs',
        '--two-stems=vocals',  # Separate into vocals and background
        input_audio_path,
        '-o', output_directory
    ]
    subprocess.run(command, check=True)
    # print(f"Demucs separation complete. Check '{output_directory}' for results.")
    log.debug("SEPARATE AUDIO - COMPLETE")


def learn_tts() -> TTS:
    # u can use tortoise-tts
    log.debug("INIT TTS MODEL - PROCESS")
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # List available 🐸TTS models
    print(TTS().list_models())
    # Init TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    # tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=True, gpu=False)
    # Fine-tune or clone the voice (this is typically more complex and may require scripts from Coqui's GitHub)
    # For demo purposes, you can synthesize with a pre-trained voice
    log.debug("INIT TTS MODEL - COMPLETE")
    return tts


# Step 2: Transcribe audio with timestamps using Whisper
def transcribe_audio_with_timestamps(audio_path):
    log.debug("TRANSCRIBE AUDIO - PROCESS")
    model = whisper.load_model("base")  # Choose model size: 'tiny', 'base', 'small', etc.
    result = model.transcribe(audio_path, fp16=False)

    segments = []
    for segment in result['segments']:
        text = segment['text'].strip()  # Clean text
        start_time = segment['start']
        end_time = segment['end']
        segments.append((text, start_time, end_time))

    log.debug("TRANSCRIBE AUDIO - COMPLETE")
    return segments


# Step 3: Translate each segment using deep-translator
def translate_segments(segments, src_lang, dest_lang):
    log.debug("TRANSLATE - PROCESS")
    translated_segments = []
    translator = GoogleTranslator(source=src_lang, target=dest_lang)

    for text, start, end in segments:
        translated_text = translator.translate(text)
        translated_segments.append((translated_text, start, end))

    log.debug("TRANSLATE - COMPLETE")
    return translated_segments


# Step 4: Synthesize speech for each segment
def synthesize_segments(translated_segments, lang_code, tts: TTS, voice_file_path):
    log.debug("SYNTHESIZE - PROCESS")
    audio_segments = []

    for i, (text, start, end) in enumerate(translated_segments):
        # tts = gTTS(text, lang=lang_code)
        output_path = f'segment_{i}.mp3'
        # tts.save(output_path)
        tts.tts_to_file(text=text, speaker_wav=voice_file_path, language=lang_code, file_path=output_path)
        audio_segments.append((output_path, start, end))
    log.debug("SYNTHESIZE - COMPLETE")
    return audio_segments


# Step 5: Combine audio segments with background sound
def combine_audio_segments_with_background(audio_segments, background_audio_path):
    def detect_leading_silence(sound, silence_threshold=-40.0, chunk_size=10):
        """
        sound is a pydub.AudioSegment
        silence_threshold in dB
        chunk_size in ms

        iterate over chunks until you find the first one with sound
        """
        trim_ms = 0  # ms

        assert chunk_size > 0  # to avoid infinite loop
        while sound[trim_ms:trim_ms + chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
            trim_ms += chunk_size

        return trim_ms

    log.debug("BUILD FINAL AUDIO - PROCESS")
    background_audio = AudioSegment.from_file(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                           background_audio_path))
    final_audio = background_audio

    for audio_path, start, end in audio_segments:
        segment_duration = (end - start) * 1000  # Duration in milliseconds
        stretched_audio = AudioSegment.from_file(audio_path)
        start_trim = detect_leading_silence(stretched_audio)
        end_trim = detect_leading_silence(stretched_audio.reverse())

        duration = len(stretched_audio)
        stretched_audio = stretched_audio[start_trim:duration - end_trim]
        # Stretch the audio to the desired segment duration
        print(len(stretched_audio))
        print(segment_duration)
        if len(stretched_audio) > segment_duration:
            stretched_audio = speedup(stretched_audio, playback_speed=len(stretched_audio) / segment_duration)

        # Overlay the translated audio onto the background
        final_audio = final_audio.overlay(stretched_audio, position=start * 1000)

    output_path = "final_audio_with_background.wav"
    final_audio.export(output_path, format="wav")
    log.debug("BUILD FINAL AUDIO - COMPLETE")
    return output_path


# Step 6: Replace original audio in video
def replace_audio_in_video(video_path, new_audio_path, output_video_path):
    log.debug("REPLACE AUDIO - PROCESS")
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(new_audio_path)
    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
    log.debug("REPLACE AUDIO - COMPLETE")


# Main function to run all steps
def main():
    video_path = 'input_video.mp4'
    audio_output_path = 'extracted_audio.wav'
    output_video_path = 'translated_video.mp4'
    output_directory = 'demucs_output'
    src_language = 'ru'  # Source language code
    dest_language = 'en'  # Target language code

    # Step 1: Extract audio from video
    extract_audio(video_path, audio_output_path)
    run_demucs(audio_output_path, output_directory)

    # Path to the accompaniment (no vocals)
    song_name = os.path.splitext(os.path.basename(audio_output_path))[0]
    background_audio_path = os.path.join(output_directory, "htdemucs", f"{song_name}/no_vocals.wav")
    voices_audio_path = os.path.join(output_directory, "htdemucs", f"{song_name}/vocals.wav")
    # Step 2: Transcribe audio and get segments
    segments = transcribe_audio_with_timestamps(audio_output_path)

    # Step 3: Translate each segment
    translated_segments = translate_segments(segments, src_language, dest_language)

    tts = learn_tts()

    # Step 4: Synthesize speech for each translated segment
    audio_segments = synthesize_segments(translated_segments, dest_language, tts, voices_audio_path)

    # Step 5: Combine synthesized audio with background sound
    final_audio_path = combine_audio_segments_with_background(audio_segments, background_audio_path)

    # Step 6: Replace original audio in the video with the new translated audio
    replace_audio_in_video(video_path, final_audio_path, output_video_path)

    # Clean up temporary files
    log.debug("CLEANUP")
    for audio_path, _, _ in audio_segments:
        os.remove(audio_path)
    os.remove(audio_output_path)
    os.remove(final_audio_path)
    shutil.rmtree(output_directory)


# Run the main function
if __name__ == "__main__":
    main()
