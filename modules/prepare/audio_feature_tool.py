import os
import time
import tensorflow as tf
from .audio_extractor import VGGish

cur_script_dir = os.path.dirname(__file__)
MODEL_DIR = os.path.join(cur_script_dir, "audio_extractor/weights")
CAP_PROP_POS_MSEC = 0
FRAMES_PER_SECOND = 1

def extract_audio_file(video_file, audio_file="", temp_audios_dir="/data02/changqing/ZyMultiModal_Data/temp_audios"):
    if not audio_file:
        base_name = os.path.basename(video_file).replace(".mp4", ".wav")
        audio_file = os.path.join(temp_audios_dir, base_name)
    # output_audio = video_file.replace('.mp4', '.wav')
    if not os.path.exists(audio_file):
        command = 'ffmpeg -loglevel error -i ' + video_file + ' ' + audio_file
        os.system(command)
        # print("audio file not exists: {}".format(output_audio))
        # return
    return audio_file

def remove_audio_file(audio_file):
    if os.path.isfile(audio_file) and os.path.exists(audio_file) and audio_file.endswith("wav"):
        os.remove(audio_file)

class AudioFeatureExtractor:
    def __init__(self):
        self.model = VGGish(MODEL_DIR)        

    def extract_features(self, video_path, temp_audios_dir="/data02/changqing/ZyMultiModal_Data/temp_audios", remove_audio=False):
        start_time = time.time()
        audio_file = extract_audio_file(video_path, temp_audios_dir=temp_audios_dir)
        audio_features = self.extract_features_from_audio(audio_file=audio_file, remove_audio=remove_audio)
        end_time = time.time()
        # print("audio extract cost {} sec".format(end_time - start_time))
        return audio_features

    def extract_features_from_audio(self, audio_file, remove_audio=True):
        audio_features = self.model(audio_file)
        if remove_audio:
            remove_audio_file(audio_file)
        return audio_features


if __name__ == '__main__':
    from glob import glob
    import numpy as np

    audio_feature_extractor = AudioFeatureExtractor()

    for audio_file in glob('./*.wav'):
        print(audio_file)
        audio_npy_path = audio_file.replace('.wav', '.npy')
        audio_feature = audio_feature_extractor.extract_features_from_audio(audio_file)
        np.save(audio_npy_path, audio_feature)
        print("Save")
