import os
import cv2
import multiprocessing as mp
from torch.nn import functional as F


def pad_image(img, padding):
    return F.pad(img, padding)

def load_model(model_dir:str):
    try:
        try:
            try:
                from model.RIFE_HDv2 import Model
                model = Model()
                model.load_model(model_dir, -1)
                print("Loaded v2.x HD model.")
            except:
                from train_log.RIFE_HDv3 import Model
                model = Model()
                model.load_model(model_dir, -1)
                print("Loaded v3.x HD model.")
        except:
            from model.RIFE_HD import Model
            model = Model()
            model.load_model(model_dir, -1)
            print("Loaded v1.x HD model")
    except:
        from model.RIFE import Model
        model = Model()
        model.load_model(model_dir, -1)
        print("Loaded ArXiv-RIFE model")
    model.eval()
    model.device()
    return model

def get_video_writer(video_path, extension, fps, size):
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_path_wo_ext, _ = os.path.splitext(video_path)
    h, w = size

    vid_out_name = '{}_{}fps.{}'.format(video_path_wo_ext, fps, extension)
    return vid_out_name, cv2.VideoWriter(vid_out_name, fourcc, fps, (w, h))

def transferAudio(sourceVideo, targetVideo, fps, bitrate):
    import shutil
    tempAudioFileName = "./temp/audio.mkv"

    # split audio from original video file and store in "temp" directory
    if True:

        # clear old "temp" directory if it exits
        if os.path.isdir("temp"):
            # remove temp directory
            shutil.rmtree("temp")
        # create new "temp" directory
        os.makedirs("temp")
        # extract audio from video
        os.system('ffmpeg -y -i "{}" -c:a copy -vn {}'.format(sourceVideo, tempAudioFileName))

    targetNoAudio = os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    os.rename(targetVideo, targetNoAudio)

    num_threads = mp.cpu_count() // 2
    # combine audio file and new video file (*.mp4)
    if os.path.exists(tempAudioFileName):
        if fps == 29.97:
            os.system('ffmpeg -y -i "{}" -i {} -c:v libx264 -b:v {} -r 29.97 -c:a aac -ar 48000 -b:a 160k -async 1 -shortest -threads {} "{}"'.format(targetNoAudio, tempAudioFileName, bitrate, num_threads, targetVideo))
        elif fps == 59.94:
            os.system('ffmpeg -y -i "{}" -i {} -c:v libx264 -b:v {} -r 59.94 -c:a aac -ar 48000 -b:a 160k -async 1 -shortest -threads {} "{}"'.format(targetNoAudio, tempAudioFileName, bitrate, num_threads, targetVideo))
        else:
            os.system('ffmpeg -y -i "{}" -i {} -c:v libx264 -b:v {} -r {}" -c:a aac -ar 48000 -b:a 160k -async 1 -shortest -threads {} "{}"'.format(targetNoAudio, tempAudioFileName, bitrate, fps, num_threads, targetVideo))
    else:
        print("This video can't extract audio...")
        if fps == 29.97:
            os.system('ffmpeg -y -i "{}" -c:v libx264 -b:v {} -filter:v "fps=fps=30000/1001" -threads {} "{}"'.format(targetNoAudio, bitrate, num_threads, targetVideo))
        elif fps == 59.94:
            os.system('ffmpeg -y -i "{}" -c:v libx264 -b:v {} -filter:v "fps=fps=60000/1001" -threads {} "{}"'.format(targetNoAudio, bitrate, num_threads, targetVideo))
        else:
            os.system('ffmpeg -y -i "{}" -c:v libx264 -b:v {} -filter:v "fps=fps={}" -threads {} "{}"'.format(targetNoAudio, bitrate, fps, num_threads, targetVideo))

    if os.path.getsize(targetVideo) == 0: # if ffmpeg failed to merge the video and audio together try converting the audio to aac
        tempAudioFileName = "./temp/audio.m4a"
        os.system('ffmpeg -y -i "{}" -c:a aac -b:a 160k -vn {}'.format(sourceVideo, tempAudioFileName))
        os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))
        if (os.path.getsize(targetVideo) == 0): # if aac is not supported by selected format
            os.rename(targetNoAudio, targetVideo)
            print("Audio transfer failed. Interpolated video will have no audio")
        else:
            print("Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead.")

            # remove audio-less video
            os.remove(targetNoAudio)
    else:
        os.remove(targetNoAudio)

    # remove temp directory
    shutil.rmtree("temp")
