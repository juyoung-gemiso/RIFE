import os
import av
import time
import math
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import warnings
import _thread
from utils import *
from ffmpeg_utils import *
from queue import Queue

warnings.filterwarnings("ignore")


class RIFE:
    def __init__(self, model_dir, output_base_path, debug=False):
        self.debug = debug
        self.output_base_path = output_base_path
        self.idx = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        self.model = load_model(model_dir)
    
    def _get_padding_image(self, image, padding):
        image = torch.from_numpy(np.transpose(image, (2,0,1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.
        return pad_image(image, padding)

    def build_read_buffer(self, read_buffer, videoCapture, img_dir):
        try:
            while True:
                read_buffer.put(read_image(os.path.join(img_dir, videoCapture.pop(0))))
        except:
            pass
        read_buffer.put(None)

    @torch.no_grad()
    def make_inference(self, I0, I1, n, scale):
        middle = self.model.inference(I0, I1, scale)
        if n == 1:
            return [middle]
        first_half = self.make_inference(I0, middle, n=n//2, scale=scale)
        second_half = self.make_inference(middle, I1, n=n//2, scale=scale)
        if n % 2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]
    
    def interpolate(self, one_second_frames, interval, padding, exp, scale, deleted_one_frame_or_not, delete_frame_flag, output_dir, pbar, input_fps_2x, interpolated_total_frame_count, size):
        h, w = size
        interpolated_frame_count = 0
        original_frame_start_index = 0
        for inserted_index in range(interval, len(one_second_frames), interval):
            left_frame = self._get_padding_image(one_second_frames[inserted_index - 1], padding)
            right_frame = self._get_padding_image(one_second_frames[inserted_index], padding)
            output = self.make_inference(left_frame, right_frame, exp, scale)
            if deleted_one_frame_or_not:
                if delete_frame_flag:
                    exp -= 1
                else:
                    exp += 1
                delete_frame_flag = not delete_frame_flag
            # insert original frames
            for original_frame_index in range(original_frame_start_index, inserted_index):
                pil_img = Image.fromarray(one_second_frames[original_frame_index][:, :, ::-1], mode='RGB')
                pil_img.save(f"{output_dir}/{self.idx:06d}.tga")
                self.idx += 1
                interpolated_total_frame_count += 1
            original_frame_start_index = inserted_index
            # insert interpolated frames
            for mid in output:
                mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
                pil_img = Image.fromarray(np.ascontiguousarray(mid[:h, :w])[:, :, ::-1], mode='RGB')
                pil_img.save(f"{output_dir}/{self.idx:06d}.tga")
                self.idx += 1
                interpolated_total_frame_count += 1
                if self.debug: interpolated_frame_count += 1
        if self.debug: print(f"interpolated_frames count: {interpolated_frame_count}")

        pbar.update(round(input_fps_2x))
        return one_second_frames[inserted_index:], interpolated_total_frame_count

    def run(self, video:str, extension:str, output_fps:float, bitrate, scale:float=1.0):
        if self.debug:
            start_time = time.time()

        # -- get metadata
        videoCapture = cv2.VideoCapture(video)
        input_fps = videoCapture.get(cv2.CAP_PROP_FPS)
        input_fps_2x = input_fps * 2
        videoCapture.release()
        first_frame = next(av.open(video).decode(video=0))
        if first_frame.interlaced_frame:
            # -- extract progressive frames from interlaced video using ffmpeg
            frames_dir = interlaced_to_progressive_2x(video, self.output_base_path, self.debug)
            frames = os.listdir(frames_dir)
            output_dir = frames_dir + "_interpolated"
        else:
            base_path, ext = os.path.splitext(os.path.basename(video))
            output_dir = os.path.join(self.output_base_path, base_path + "_interpolated")
            input_fps_2x = input_fps
        os.makedirs(output_dir, exist_ok=True)
        if self.debug:
            end_time = time.time()
            print(f"=> extract progressive frames time: {end_time - start_time}s")
            start_time = time.time()

        interval = max(1, int(input_fps_2x // (round(output_fps * 2) - round(input_fps_2x))))
        exp = math.floor((output_fps * 2) / input_fps_2x)
        deleted_one_frame_or_not = False # e.g. 1, 1, 1, 1, ...
        if (output_fps * 2) % input_fps_2x != 0 and exp > 1:
            deleted_one_frame_or_not = True # e.g. 2, 1, 2, 1, ...
        total_frames = len(frames)
        if self.debug: print(f"original fps: {input_fps_2x}, total frames: {total_frames}")
        lastframe = read_image(os.path.join(frames_dir, frames.pop(0)))
        h, w, _ = lastframe.shape
        assert int((output_fps * 2) // input_fps_2x) > 0, 'You can\'t drop frames!! You must be input to fps greater than original fps!'

        tmp = max(32, int(32 / scale))
        ph = ((h - 1) // tmp + 1) * tmp
        pw = ((w - 1) // tmp + 1) * tmp
        if self.debug: print(f"Padding Height: {ph}, Padding Width: {pw}")
        padding = (0, pw - w, 0, ph - h)
        pbar = tqdm(total=total_frames)

        # -- generate buffer
        read_buffer = Queue(maxsize=500)
        _thread.start_new_thread(self.build_read_buffer, (read_buffer, frames, frames_dir))

        # -- interpolate extracted progressive frames using RIFE
        one_second_frames = [lastframe] # length = origin fps + 1
        interpolated_total_frame_count = 0
        original_total_frame_count = 1
        delete_frame_flag = False

        while True:
            if len(one_second_frames) <= round(input_fps_2x):
                frame = read_buffer.get()
                if frame is None:
                    if one_second_frames:
                        one_second_frames, interpolated_total_frame_count = self.interpolate(
                            one_second_frames, 
                            interval, 
                            padding, 
                            exp,
                            scale, 
                            deleted_one_frame_or_not, 
                            delete_frame_flag, 
                            output_dir, 
                            pbar, 
                            input_fps_2x, 
                            interpolated_total_frame_count,
                            (h, w)
                        )               
                    break
                one_second_frames.append(frame)
                original_total_frame_count += 1
                continue
            if self.debug: 
                print(f"\none_second_frames count: {len(one_second_frames)}")
                print("=> interpolation")
            one_second_frames, interpolated_total_frame_count = self.interpolate(
                one_second_frames, 
                interval, 
                padding, 
                exp, 
                scale, 
                deleted_one_frame_or_not, 
                delete_frame_flag, 
                output_dir, 
                pbar, 
                input_fps_2x, 
                interpolated_total_frame_count,
                (h, w)
            )

        if one_second_frames:
            for frame in one_second_frames:
                pil_img = Image.fromarray(frame, mode='RGB')
                pil_img.save(f"{output_dir}/{self.idx:06d}.tga")
                self.idx += 1
                interpolated_total_frame_count += 1

        if self.debug:
            print(f"\ninterpolated_total_frame_count: {interpolated_total_frame_count}")
            print(f"original_total_frame_count: {original_total_frame_count}")
        pbar.close()

        if self.debug:
            end_time = time.time()
            print(f"=> interpolation time: {end_time - start_time}s")
            start_time = time.time()

        transferAudioWithFrames(video, output_dir, output_fps, bitrate, extension)

        if self.debug:
            end_time = time.time()
            print(f"=> total spent time: {end_time - start_time}s")
        self.idx = 0

        shutil.rmtree(frames_dir)
        shutil.rmtree(output_dir)

if __name__ == '__main__':
    r"""
    Example Command Line:

        > python inference_v5.py --video=Z:\\AI_workspace\\video_interpolation_backup\\soccer_25fps_1m30s.mxf --output_base_path=Z:\\AI_workspace\\video_interpolation_backup --fps=29.97 --bitrate=60M --ext=mxf
    """
    parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
    parser.add_argument('--video', dest='video', type=str, default=None)
    parser.add_argument('--output_base_path', dest='output_base_path', type=str, default=None)
    parser.add_argument('--fps', dest='fps', type=float, default=None)
    parser.add_argument('--bitrate', dest='bitrate', type=str, default=None, help="e.g. 60M")
    parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='vid_out video extension')
    parser.add_argument('--debug', dest='debug', action='store_true', help='whether debug or not')
    args = parser.parse_args()
    
    rife = RIFE(model_dir="train_log", output_base_path=args.output_base_path, debug=args.debug)
    rife.run(args.video, args.ext, args.fps, args.bitrate, scale=0.25)
