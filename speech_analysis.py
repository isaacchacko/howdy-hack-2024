from moviepy.editor import AudioFileClip
import speech_recognition as sr
import re
from collections import Counter

## pulling audio ##

def extract_audio(wav_file):
    audio = AudioFileClip(wav_file)
    return audio.duration

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return 'Audio Unrecognizable'
    except sr.RequestError:
        return 'API Unavailable'

## working with stuttering and filler words

def analyze_stuttering_and_filler(transcribed_text):
    words = transcribed_text.split()
    repeated_words = [word for word, count in Counter(words).items() if count > 1]
    
    # Optional: Use regex to find repeated syllables (naïve approach)
    stutters = re.findall(r'\b(\w+)\s+\1\b', transcribed_text)

    fillers = ['uh', 'um', 'oh', 'like', 'you know', 'so']
    filler_counts = {filler: transcribed_text.lower().count(filler) for filler in fillers}

    return {
        "repeated_words": repeated_words,
        "stutters": stutters,
        "filler_counts": filler_counts
    }

def output_stuttering(transcribed_text):
    stutter_analysis = analyze_stuttering_and_filler(transcribed_text)
    length_stutter = len(stutter_analysis['repeated_words'])
    if length_stutter > 0:
        if length_stutter == 1:
            return f"You stuttered on the word '{stutter_analysis['repeated_words'][0]}'" +\
                  ". However, practicing mindfulness and slowing down your speech can " +\
                  "help you gain confidence and improve fluency over time"
        else:
            words_string = ''
            for index, word in enumerate(stutter_analysis['repeated_words']):
                if index == 0:
                    words_string += f"'{word}'"
                elif index == length_stutter - 1:
                    words_string += f", and '{word}'"
                else:
                    words_string += f", '{word}'"
                
            return f"You stuttered on the words {words_string}. " + \
                   "However, practicing mindfulness and slowing down " + \
                  "your speech can help you gain confidence and improve " +\
                  "fluency over time"
    else:
        return "You didn't stutter!"

def output_filler(transcribed_text):
    stutter_analysis = analyze_stuttering_and_filler(transcribed_text)
    filler_words = []
    filler_counts_dict = stutter_analysis['filler_counts']
    for key, value in filler_counts_dict.items():
        if value > 0:
            filler_words.append([key, value])
    length_filler = len(filler_words)
    if length_filler > 0:
        if length_filler == 1:
            get_first_val = filler_words[0][1]
            if get_first_val == 1:
                return f"You said the filler word '{filler_words[0][0]}'" + \
                        f" 1 time. Reducing filler words takes practice; " +\
                "try pausing briefly when you feel the urge to use them—this gives " + \
            "you time to think and enhances your clarity!"
            else:
                return f"You said the filler word '{filler_words[0][0]}'" + \
                        f" {filler_words[0][1]} times. Reducing filler words takes practice; " +\
                "try pausing briefly when you feel the urge to use them—this gives " + \
            "you time to think and enhances your clarity!"
        else:
            words_string = ''
            for sublist in filler_words:
                words_string += f"\n'{sublist[0]}': {sublist[1]}"
            
            words_string += "\nReducing filler words takes practice; " +\
                "try pausing briefly when you feel the urge to use them—this gives " + \
            "you time to think and enhances your clarity!"
                
            return f"You said the following filler words: {words_string}. "
    else:
        return "You didn't say any filler words!"
    
## working with wpm ##

def calculate_wpm(transcribed_text, duration):
    word_count = len(transcribed_text.split())
    minutes = duration / 60
    wpm = word_count / minutes if minutes > 0 else 0
    return wpm

def output_wpm(wpm):
    if wpm > 150:
        return f'You spoke at {round(wpm)} wpm. Slow down a bit to give your audience ' + \
              'time to absorb your enthusiasm—pauses can enhance your message!'
    elif wpm < 100:
        return f'You spoke at {round(wpm)} wpm. Add a bit more energy to your speech by ' + \
              'practicing with a metronome to find a comfortable pace!'
    else:
        return f'You spoke at {round(wpm)}. Keep up the good pace!'

def main(wav_file):

    # Extract audio
    audio_duration = extract_audio(wav_file)

    # Transcribe audio
    transcribed_text = transcribe_audio(wav_file)
    print('Transcribed Text:', transcribed_text)

    # Analyze speech
    print(output_stuttering(transcribed_text))

    # get wpm
    wpm = calculate_wpm(transcribed_text, audio_duration)
    print(output_wpm(wpm))

    # output the filller words
    print(output_filler(transcribed_text))


main('extracted_audio.wav')