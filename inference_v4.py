import os
import cv2
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
from queue import Queue

warnings.filterwarnings("ignore")

class RIFE:
    def __init__(self, model_dir, debug=False, save_img=False):
        self.debug = debug
        self.img_dir = "uniform_distributed_imgs"
        self.save_img = save_img
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

    def clear_write_buffer(self, write_buffer):
        while True:
            item = write_buffer.get()
            if item is None:
                break
            self.vid_out.write(item)

    def build_read_buffer(self, read_buffer, videoCapture):
        try:
            while videoCapture.isOpened():
                read_buffer.put(videoCapture.read()[1])
        except:
            pass
        read_buffer.put(None)
        videoCapture.release()

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

    def run(self, video:str, extension:str, fps:float|int, bitrate, scale:float=1.0):
        if self.debug:
            start_time = time.time()
        videoCapture = cv2.VideoCapture(video)
        original_fps = videoCapture.get(cv2.CAP_PROP_FPS)
        interval = max(1, int(original_fps // (round(fps) - round(original_fps))))
        exp = math.floor(fps / original_fps)
        deleted_one_frame_or_not = False # e.g. 1, 1, 1, 1, ...
        if fps % original_fps != 0 and exp > 1:
            deleted_one_frame_or_not = True # e.g. 2, 1, 2, 1, ...
        total_frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        if self.debug:
            print(f"original fps: {original_fps}, total frames: {total_frames}")

        lastframe = videoCapture.read()[1]
        h, w, _ = lastframe.shape
        if int(fps // original_fps) == 0:
            print('You can\'t drop frames!! You must be input to fps greater than original fps!')
            exit(0)

        vid_out_name, self.vid_out = get_video_writer(video, extension, fps, (h, w))
        
        tmp = max(32, int(32 / scale))
        ph = ((h - 1) // tmp + 1) * tmp
        pw = ((w - 1) // tmp + 1) * tmp
        if self.debug:
            print(f"Padding Height: {ph}, Padding Width: {pw}")
        padding = (0, pw - w, 0, ph - h)
        pbar = tqdm(total=total_frames)

        write_buffer = Queue(maxsize=500)
        read_buffer = Queue(maxsize=500)
        _thread.start_new_thread(self.build_read_buffer, (read_buffer, videoCapture))
        _thread.start_new_thread(self.clear_write_buffer, (write_buffer,))

        one_second_frames = [lastframe] # length = origin fps + 1
        interpolated_total_frame_count = 0
        original_total_frame_count = 1
        delete_frame_flag = False

        while True:
            if len(one_second_frames) <= round(original_fps):
                frame = read_buffer.get()
                if frame is None:
                    break
                one_second_frames.append(frame)
                original_total_frame_count += 1
                continue
            if args.debug: print(f"\none_second_frames count: {len(one_second_frames)}")
            # -- interpolation
            if args.debug: print("=> interpolation")
            interpolated_frame_count = 0
            original_frame_start_index = 0
            for inserted_index in range(interval, round(original_fps) + 1, interval):
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
                    if self.save_img:
                        pil_img = Image.fromarray(one_second_frames[original_frame_index][:, :, ::-1], mode='RGB')
                        pil_img.save(f"{self.img_dir}/{self.idx}.tga")
                        # cv2.imwrite(f"{self.img_dir}/{self.idx}.tga", one_second_frames[original_frame_index])
                        self.idx += 1
                    write_buffer.put(one_second_frames[original_frame_index])
                    interpolated_total_frame_count += 1
                original_frame_start_index = inserted_index
                # insert interpolated frames
                for mid in output:
                    mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
                    if self.save_img:
                        pil_img = Image.fromarray(np.ascontiguousarray(mid[:h, :w])[:, :, ::-1], mode='RGB')
                        pil_img.save(f"{self.img_dir}/{self.idx}_interpolated.tga")
                        # cv2.imwrite(f"{self.img_dir}/{self.idx}_interpolated.tga", np.ascontiguousarray(mid[:h, :w]))
                        self.idx += 1
                    write_buffer.put(np.ascontiguousarray(mid[:h, :w]))
                    interpolated_total_frame_count += 1
                    if args.debug: interpolated_frame_count += 1
            if args.debug: print(f"interpolated_frames count: {interpolated_frame_count}")

            pbar.update(round(original_fps))
            one_second_frames = one_second_frames[-1:]

        if one_second_frames:
            for frame in one_second_frames:
                write_buffer.put(frame)
                interpolated_total_frame_count += 1
        write_buffer.put(None)

        if self.debug:
            print(f"\ninterpolated_total_frame_count: {interpolated_total_frame_count}")
            print(f"original_total_frame_count: {original_total_frame_count}")

        while(not write_buffer.empty()):
            time.sleep(0.1)
        pbar.close()
        if not self.vid_out is None:
            self.vid_out.release()

        # move audio to new video file if appropriate
        try:
            transferAudio(video, vid_out_name, fps, bitrate)
        except:
            print("Audio transfer failed. Interpolated video will have no audio")
            targetNoAudio = os.path.splitext(vid_out_name)[0] + "_noaudio" + os.path.splitext(vid_out_name)[1]
            os.rename(targetNoAudio, vid_out_name)

        if self.debug:
            end_time = time.time()
            print(f"=> total spent time: {end_time - start_time}s")
        if self.save_img: self.idx = 0

if __name__ == '__main__':
    r"""
    Example Command Line:

        > python inference_v4.py --video=C:\Users\gemiso\Desktop\frame_interpol_videos\\son_goal_01_25fps.mp4 --fps=29.97 --bitrate=60M
    """
    parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
    parser.add_argument('--video', dest='video', type=str, default=None)
    parser.add_argument('--output', dest='output', type=str, default=None)
    parser.add_argument('--fps', dest='fps', type=float, default=None)
    parser.add_argument('--bitrate', dest='bitrate', type=str, default=None, help="e.g. 60M")
    parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='vid_out video extension')
    parser.add_argument('--debug', dest='debug', action='store_true', help='whether debug or not')
    parser.add_argument('--save-img', dest='save_img', action='store_true', help='whether debug or not')
    args = parser.parse_args()
    
    rife = RIFE(model_dir="train_log", debug=args.debug, save_img=args.save_img)
    rife.run(args.video, args.ext, args.fps, args.bitrate, scale=0.25)
