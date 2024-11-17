from deep_translator import GoogleTranslator
from moviepy.editor import VideoFileClip, AudioFileClip
import whisper
from gtts import gTTS
from pydub import AudioSegment
import os

from pydub.effects import speedup


# Step 1: Extract audio from video
def extract_audio(video_path, audio_output_path):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_output_path)
    video.close()


# Step 2: Transcribe audio with timestamps using Whisper
def transcribe_audio_with_timestamps(audio_path):
    model = whisper.load_model("base")  # You can use 'tiny', 'base', 'small', 'medium', 'large'
    result = model.transcribe(audio_path, fp16=False)

    segments = []
    for segment in result['segments']:
        word = segment['text']
        start_time = segment['start']
        end_time = segment['end']
        segments.append((word, start_time, end_time))

    return segments


# Step 3: Translate each segment using deep-translator
def translate_segments(segments, src_lang, dest_lang):
    translated_segments = []
    translator = GoogleTranslator(source=src_lang, target=dest_lang)

    for word, start, end in segments:
        translated_text = translator.translate(word)
        translated_segments.append((translated_text, start, end))

    return translated_segments


# Step 4: Synthesize speech for each segment
def synthesize_segments(translated_segments, lang_code):
    audio_segments = []

    for i, (text, start, end) in enumerate(translated_segments):
        tts = gTTS(text, lang=lang_code, slow=False)
        output_path = f'segment_{i}.mp3'
        tts.save(output_path)
        audio_segments.append((output_path, start, end))

    return audio_segments


# Step 5: Combine audio segments with timing alignment
def combine_audio_segments(audio_segments):
    final_audio = AudioSegment.silent(duration=0)

    for audio_path, start, end in audio_segments:
        segment_audio = AudioSegment.from_file(audio_path)
        segment_audio = speedup(segment_audio, playback_speed=1.3)
        segment_duration = (end - start) * 1000  # Convert to milliseconds
        silence_duration = max(0, segment_duration - len(segment_audio))
        segment_audio = segment_audio + AudioSegment.silent(duration=silence_duration)
        final_audio += segment_audio

    output_path = "final_audio.mp3"
    final_audio.export(output_path, format="mp3")
    return output_path


# Step 6: Replace original audio in video
def replace_audio_in_video(video_path, new_audio_path, output_video_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(new_audio_path)
    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac")


# Main function to run all steps
def main():
    video_path = 'input_video.mp4'
    audio_output_path = 'extracted_audio.wav'
    output_video_path = 'translated_video.mp4'
    src_language = 'ru'  # Source language code
    dest_language = 'en'  # Target language code
    tts_language_code = 'en'  # TTS language code for target language

    # Step 1: Extract audio from video
    extract_audio(video_path, audio_output_path)

    # Step 2: Transcribe audio and get segments
    segments = transcribe_audio_with_timestamps(audio_output_path)

    # Step 3: Translate each segment
    translated_segments = translate_segments(segments, src_language, dest_language)

    # Step 4: Synthesize speech for each translated segment
    audio_segments = synthesize_segments(translated_segments, tts_language_code)

    # Step 5: Combine audio segments and align with original timing
    final_audio_path = combine_audio_segments(audio_segments)

    # Step 6: Replace original audio in the video with the new translated audio
    replace_audio_in_video(video_path, final_audio_path, output_video_path)

    # Clean up temporary files
    for audio_path, _, _ in audio_segments:
        os.remove(audio_path)


# Run the main function
if __name__ == "__main__":
    main()
