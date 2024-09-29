
# transcript
import whisper
import ffmpeg

# ppt
import re
import zipfile
import os

import cv2
import shutil

# for fuzzy search
import nltk
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import re
from collections import Counter

# updating json
import json
import numpy

def get_frame_rate(video_file):
    try:
        # Use ffmpeg.probe to retrieve metadata about the video file
        probe = ffmpeg.probe(video_file)
        
        # Extract the video stream information
        video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
        
        if video_streams:
            # Get the frame rate from the first video stream
            frame_rate = eval(video_streams[0]['r_frame_rate'])  # e.g., "30/1"
            return frame_rate
        else:
            print("No video streams found.")
            return None
    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
    except Exception as e:
        print(f"An error occurred: {e}")

def detect_faces_eyes_from_frame(frame, draw=False):
    
    '''
    input:
        frame: image

    output:
        int, number of faces detected
    '''
    
    # return variable
    count = 0

    # Load Haar cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

    # Read an image or video frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop over detected faces
    for (x, y, w, h) in faces:

        if draw: cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Focus on the face region
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Check if two eyes are detected
        if len(eyes) == 2:

            if draw:
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
            count += 1  # increase count for a successful face+2eyes detection
    
    return count, frame

def get_retention(filename, folder=None, step_through=False, debug=False):
    
    '''
    input:
        filename: string of video file, name + extension
        folder: string of folder name.
                when set, individual marked frames will be saved here.
        step_through: boolean. when set, cv2 will open a marked image
                      preview window.
        debug: boolean. when set, retention values will be printed as
               well as returned.

    output:
        [int, int, int, ...] list of retention values

    '''
    cap = cv2.VideoCapture(filename)
    count = 0
    retention_values = []

    status, frame = cap.read()
    while status:
        # if user requested to save all frames to folder
        if folder: cv2.imwrite(os.path.join(folder, "frame%d.jpg" % count), frame)

        # the meat
        retention, marked_frame = detect_faces_eyes_from_frame(frame, draw=(folder != None or step_through))
        retention_values.append(retention)
     
        # if we're in step_through mode
        if step_through:   
            cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
            cv2.imshow('frame-by-frame viewer: press n', marked_frame)
            print(f'Frame Count: {count}, Retention: {retention}')
            while cv2.waitKey(10) & 0xFF != ord('n'):
                pass 
                
        count = count + 1
        status, frame = cap.read()
    
    # if image previews were opened (only possible in step_through mode) 
    if step_through:
        cap.release()
        cv2.destroyAllWindows() # destroy all opened windows
    
    if debug:
        for i in retention_values:
            print(i, end=', ')

    return retention_values

def clean_up(folderpath):
    shutil.rmtree(folderpath)

def get_text(filename, debug=False):
        
    '''
    input:
        filename: string of powerpoint
        debug: boolean. if set, will print debug information

    output:
        [str, str, ...] - list of strings representing a list
                          of text from each slide
    '''
    
    if '.pptx' not in filename:
        raise Exception("Invalid filetype .{filename.split('.')[-1]} supported. Only .pptx please!")
    # return variable
    slide_script = []

    try:
        # assuming user inputted a pptx
        filepath = os.path.join(os.getcwd(), filename)
        shutil.copy(filepath, filepath+'.zip')

        # make folder by taking filename (split/cut off file extension)
        folder = os.path.basename(filepath).split('.')[0]

        # if folder exists, delete it
        if os.path.exists(folder): shutil.rmtree(folder)

        # os.makedirs(folder)
        folderpath = os.path.join(os.getcwd(), folder)
        
        # extract files
        with zipfile.ZipFile(os.path.join(os.getcwd(), filepath+'.zip'), 'r') as zip_ref:
            zip_ref.extractall(folderpath)

        slidespath = os.path.join(folderpath, 'ppt', 'slides')
        filenames = [file for file in os.listdir(slidespath) if '.' in file and 'xml' in file]
        sorted_filenames = sorted(filenames, key=lambda f: int(re.search(r'\d+', f).group()))
        
        for file in sorted_filenames:
            with open(os.path.join(slidespath, file), 'r') as f:
                content = f.read()
            pattern = r'<a:t>(.*?)</a:t>'
            matches = re.findall(pattern, content)
            slide_script.append(matches)
            if debug: print(f'{file = } {len(matches) = }')
        
            # i believe that only xml files can exist here
            # this is an warning
            if 'xml' not in file and debug:
                print(f'WARNING: non-xml file found: {file}')
        
        return slide_script

    except Exception as e:
        raise e

def get_transcript(filename):
    
    '''
    input:
        filename of video
    
    output:
        list of segments formatted as the following:
        [[(float start, float end), 
           string text], ...]
        
    '''

    # return variables
    segs = []

    video_filepath = os.path.join(os.getcwd(), filename)
    audio_filepath = os.path.join(os.getcwd(), 'tmp.wav')
    ffmpeg.input(video_filepath).output(audio_filepath).run(overwrite_output=True)

    model = whisper.load_model("base")
    result = model.transcribe(audio_filepath)
    for seg in result['segments']:
        segs.append([(seg['start'], seg['end']), seg['text']])

    return segs

def find_unique(slide_text_list):

    '''
    input:
        slide_text_list : [[str, str, str, ...], ...]
            each index is a new slide
            strings are either words or phrases

    output:
        unique_words : [[str, str, str, ...], ...]
            each index is a new slide
            strings are single words

    description:
        parses through slide_text_list and finds unique words on each slide

    '''
    
    # return variable
    unique_words = []
    
    # input data massaging
    # new form : [str, str, ...]
    #            new words are separated by space
    
    slide_text_word_list = [re.sub(r'[^a-zA-Z ]', '', ' '.join(i).lower()) for i in slide_text_list] 
    
    # counts the number of each word
    word_count_per_slide = []
    for slide_words in slide_text_word_list:
        words = re.findall(r'\b\w+\b', slide_words.lower())
        word_count_per_slide.append(Counter(words))
    
    global_word_count = sum(word_count_per_slide, start=Counter()) 

    for slide_index, words in enumerate(slide_text_word_list):
        tmp_list = []
        for word in words.split():
            # checks if there's either only one count of the word OR
            # if all the counts of the word are in the same line
            if 1 == global_word_count[word] or global_word_count[word] == word_count_per_slide[slide_index][word]:
                tagged = nltk.pos_tag([word])
                if tagged[0][1] in ['NN', 'NNS', 'NNP', 'NNPS']:
                    tmp_list.append(word)
            
        unique_words.append(tmp_list)
    
    return unique_words

def fuzzy_search(slide_text_list, transcript_text):

    '''
    input:
        slide_text_list : [[str, str, str, ...], ...]
            each index is a new slide
            strings are either words or phrases
        transcript_text: 
            [[(float start, float end), 
               string text], ...]
    
    output:
        [[start : float, slide_index : int], ...]
        meant to be traversed by min bound start until satisfied
    
    description:
        uses a fuzzy search to find similar words
        
    '''

    # return variable
    time_to_slide_index = []

    unique_words = find_unique(slide_text_list)

    combined_unique = {}
    for slide_index, slide_words in enumerate(unique_words):
        for word in slide_words:
            combined_unique[word] = slide_index

    # combined_unique = [item for sublist in unique_words for item in sublist]
    for index, transcript_info in enumerate(transcript_text):
        line_by_words = []
        for word in transcript_info[1].split(' '):
            word = re.sub(r'[^a-zA-Z]', '', word.lower())
            if word != '':
                result = process.extractOne(word, list(combined_unique.keys()), score_cutoff=95)
                if result != None:
                    best_match, score = result
                    # print(f'transcript_line_index:{index},start:{transcript_info[0][0]},word:{word},best_match:{best_match},score:{score},slide:{combined_unique[best_match]}')
                    line_by_words.append(combined_unique[best_match])
        
        if len(line_by_words) > 0:
            # raise Exception(f'ERROR: No words in line of transcript "{transcript_info[2]}" matched with any slide content.')
            calculated_slide_index, freq = Counter(line_by_words).most_common(1)[0]
            if len(time_to_slide_index) == 0:
                last_index = 0
            else:
                last_index = time_to_slide_index[-1][1]
            if abs(calculated_slide_index - last_index) <= 1:
                time_to_slide_index.append([transcript_info[0][0], calculated_slide_index]) 

    return time_to_slide_index

# print(find_unique(get_text('test_ppt.pptx')))
# transcript = get_transcript('test_transcript.mp4')
# print(fuzzy_search(get_text('test_ppt.pptx'), transcript))


## JUSTIN'S SECTION ##
import os
# from pptx import Presentation
# from pdf2image import convert_from_path
# from tempfile import NamedTemporaryFile
# from pathlib import Path

def pptx_to_png(pptx_path, output_dir):
    # Load the PowerPoint presentation
    presentation = Presentation(pptx_path)

    # Create a temporary PDF file
    with NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        temp_pdf_path = temp_pdf.name
        # Save the PowerPoint as PDF
        presentation.save(temp_pdf_path)

    # Convert PDF to images
    images = convert_from_path(temp_pdf_path)

    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save each slide as PNG
    png_files = []
    for i, image in enumerate(images):
        png_file = os.path.join(output_dir, f"slide{i + 1}.png")
        image.save(png_file, 'PNG')
        png_files.append(png_file)

    # Clean up the temporary PDF file
    os.remove(temp_pdf_path)

    return png_files

# Example usage
# pptx_path = 'temp_ppt.pptx'  # Replace with your PowerPoint file path
# output_dir = 'new_frontend/public/path/to'            # Specify your desired output directory
# png_files = pptx_to_png(pptx_path, output_dir)
# print("Converted PNG files:", png_files)

def main(powerpoint_filepath, video_filepath):

    transcript = get_transcript(video_filepath)
    time_to_slide_index = fuzzy_search(get_text(powerpoint_filepath, transcript), transcript)
    with open(os.path.join('backend', 'data', 'slides_data.json'), 'r') as f:
        data = json.loads(f.read())

    retention = get_retention(video_filepath, folder=None, step_through=False, debug=False)
    
    frame_rate = get_frame_rate(video_filepath)
    time = list(numpy.arange(len(retention)) / frame_rate)
    slide_changes = []
    last_slide = 0
    for start, index in time_to_slide_index:
        if index != last_slide:
            last_slide = index
            slide_changes.append(start)
    
    data['retention'] = retention
    data['time'] = time
    data['slide_changes'] = slide_changes
    
    with open(os.path.join('backend', 'data', 'slides_data.json'), 'w') as f:
        f.write(json.dumps(data))

get_retention('short.mp4', step_through=True, folder='hello')
