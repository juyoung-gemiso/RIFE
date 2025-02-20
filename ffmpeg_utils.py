import os
import shutil
import multiprocessing as mp


FPS_MAPPING = {
    29.97: "30000/1001",
    59.94: "60000/1001",
}

def extract_audio(sourceVideo):
    tempAudioFileName = "./temp/audio.mkv"
     # clear old "temp" directory if it exits
    if os.path.isdir("temp"):
        # remove temp directory
        shutil.rmtree("temp")
    # create new "temp" directory
    os.makedirs("temp")
    # extract audio from video
    os.system('ffmpeg -y -i "{}" -c:a copy -vn {}'.format(sourceVideo, tempAudioFileName))
    assert os.path.exists(tempAudioFileName), "not exist audio file. maybe it didn't extracted audio."
    return tempAudioFileName

def transferAudio(sourceVideo, targetVideo, fps:float, output_ext:str, audio_stream_count:int):
    tempAudioFileName = extract_audio(sourceVideo)
    base_path, ext =  os.path.splitext(sourceVideo)
    targetNoAudio = os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    os.rename(targetVideo, targetNoAudio)
    if not output_ext.startswith('.'): output_ext = f".{output_ext}"

    num_threads = mp.cpu_count() // 2
    audio_stream_mappings = [f"-map 1:a" for _ in range(audio_stream_count)]
    # combine audio file and new video file (*.mp4)
    if output_ext.lower().endswith('mxf'):
        cli = [
            'ffmpeg -y',
            f'-i "{targetNoAudio}" -i {tempAudioFileName}',
            '-vcodec mpeg2video',
            '-s 1920x1080',
            f'-b:v 50M -minrate 50M -maxrate 50M',
            '-bufsize 36M',
            f'-r {fps}',
            '-timecode "00:00:00;00"',
            '-g 15',
            '-keyint_min 15',
            '-bf 2',
            '-pix_fmt yuv422p',
            '-flags ilme',
            '-acodec pcm_s24le -ar 48000',
            f'-filter_complex "[0:v]fps=fps={FPS_MAPPING.get(fps * 2, fps * 2)},tinterlace=mode=interlacex2,setfield=tff[v]"',
            f'-map "[v]" {" ".join(audio_stream_mappings)}',
            f'-max_muxing_queue_size 1024 -sc_threshold 1000000000 -threads {num_threads}',
            base_path + "_output" + output_ext
        ]
    
    print(f"ffmpeg command line:\n{' '.join(cli)}")
    os.system(' '.join(cli))
    os.remove(targetNoAudio)

    # remove temp directory
    shutil.rmtree("temp")

@DeprecationWarning
def transferAudioWithFrames(sourceVideo:str, targetFrames:str, fps:float, bitrate:str, output_ext:str):
    tempAudioFileName = extract_audio(sourceVideo)
    base_path, ext =  os.path.splitext(sourceVideo)
    if not output_ext.startswith('.'): output_ext = f".{output_ext}"

    num_threads = mp.cpu_count() // 2
    # merge audio and video to interlaced video(e.g. 60p -> 29.97i)
    if output_ext.lower().endswith('mxf'):
        cli = [
            'ffmpeg -y',
            f'-framerate {FPS_MAPPING.get(fps * 2, fps * 2)}',
            f'-i "{targetFrames}/%06d.tga" -i {tempAudioFileName}',
            '-vcodec mpeg2video',
            '-s 1920x1080',
            f'-b:v {bitrate} -minrate {bitrate} -maxrate {bitrate}',
            '-bufsize 100M',
            '-timecode "00:00:00;00"',
            '-g 15',
            '-keyint_min 15',
            '-bf 2',
            '-pix_fmt yuv422p',
            '-flags ilme',
            '-acodec pcm_s24le -ar 48000',
            f'-filter_complex "[0:v]fps=fps={FPS_MAPPING.get(fps * 2, fps * 2)},tinterlace=mode=interlacex2,setfield=tff[v]"',
            '-map "[v]" -map 1:a -map 1:a',
            f'-max_muxing_queue_size 1024 -sc_threshold 1000000000 -threads {num_threads}',
            base_path + "_output" + output_ext
        ]
    # merge audio and video to progressive video(e.g. )
    elif output_ext.lower().endswith('mp4'):
        cli = [
            'ffmpeg -y',
            f'-framerate {FPS_MAPPING.get(fps, fps)}',
            f'-i "{targetFrames}/%06d.tga" -i {tempAudioFileName}',
            f'-b:v {bitrate} -minrate {bitrate} -maxrate {bitrate}',
            '-map 0:v -map 0:a:0 -map 0:a:1',
            '-c:a aac -b:a 192k',
            f'-threads {num_threads}',
            base_path + "_output" + output_ext
        ]
    os.system(' '.join(cli))

    # remove temp directory
    shutil.rmtree("temp")

def get_output_frames_path(video_path:str, output_base_path:str):
    base_path, ext = os.path.splitext(os.path.basename(video_path))
    output_frames_path = os.path.join(output_base_path, base_path + "_p_2x")
    os.makedirs(output_frames_path, exist_ok=True, mode=0o777)
    return output_frames_path

def interlaced_to_progressive_2x_frames(video_path:str, output_frames_path:str, debug:bool=False) -> None:
    if debug: print(f'=> interlaced_to_progressive_2x: {video_path} -> {output_frames_path}')
    command = f'ffmpeg -i {video_path} -vf "yadiif=1" -frame_pts true {output_frames_path}/%6d.tga'
    os.system(command)
