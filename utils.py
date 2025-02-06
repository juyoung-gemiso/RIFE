import os
import av
import cv2
import _thread
import numpy as np
from PIL import Image
from queue import Queue
from torch.nn import functional as F


def pad_image(img, padding):
    return F.pad(img, padding)

def save_image(img:np.array, save_path:str):
    Image.fromarray(img, mode='RGB').save(save_path)

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

def get_video_writer_using_av(video_path, fps, size, extension="mp4"):
    video_path_wo_ext, _ = os.path.splitext(video_path)
    h, w = size
    output_video_name = '{}_{}fps.{}'.format(video_path_wo_ext, fps, extension)
    container = av.open(output_video_name, mode="w")
    stream = container.add_stream(
        codec_name="mpeg2video", 
        rate=fps, 
        options={
            'maxrate': '50M', 
            'minrate': '50M', 
            'bufsize': '36M',
        }
    )
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    stream.bit_rate = 50_000_000
    stream.gop_size = 15

    return output_video_name, container, stream

def read_image(path:str):
    return cv2.cvtColor(np.array(Image.open(path)), cv2.COLOR_BGR2RGB)

def clear_write_buffer(write_buffer:Queue, video_container:av.container.OutputContainer, video_stream:av.VideoStream, size:tuple[int, int]):
    height, width = 0, 1
    while True:
        item = np.ascontiguousarray(write_buffer.get())
        if item is None:
            break
        frame = av.VideoFrame.from_ndarray(np.frombuffer(item, dtype=np.uint8).reshape(size[height], size[width], 3), format="rgb24")
        for packet in video_stream.encode(frame):
            video_container.mux(packet)

def build_read_buffer(read_buffer, videoCapture):
    try:
        while True:
            read_buffer.put(read_image(videoCapture.pop(0)))
    except:
        pass
    read_buffer.put(None)
    
def generate_buffer(frames:list[str], video_container:av.container.OutputContainer=None, video_stream:av.VideoStream=None, size:tuple[int, int]=None):
    read_buffer = Queue(maxsize=500)
    _thread.start_new_thread(build_read_buffer, ((read_buffer, frames)))
    write_buffer = Queue(maxsize=500)
    _thread.start_new_thread(clear_write_buffer, (write_buffer, video_container, video_stream, size))
    return read_buffer, write_buffer

def frame2image(frame):
    return cv2.cvtColor(frame.to_rgb().to_ndarray(), cv2.COLOR_RGB2BGR)
