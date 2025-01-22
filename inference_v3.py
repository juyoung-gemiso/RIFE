import os
import cv2
import time
import math
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F

import warnings
import _thread
from utils import *
from queue import Queue
from model.pytorch_msssim import ssim_matlab

warnings.filterwarnings("ignore")

class RIFE:
    def __init__(self, model_dir, debug=False, save_img=False):
        self.debug = debug
        self.img_dir = "low_ssim_imgs"
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
            try:
                self.vid_out.write(item)
            except:
                import pdb; pdb.set_trace()

    def build_read_buffer(self, read_buffer, videoCapture):
        try:
            while videoCapture.isOpened():
                read_buffer.put(videoCapture.read()[1])
        except:
            pass
        read_buffer.put(None)
        videoCapture.release()
    
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

        one_second_frames = [lastframe] # length=origin fps
        temp_interpolated_frames = []
        total_interpolated_fps = 0
        original_total_frame_count = 1

        while True:
            if len(one_second_frames) <= original_fps:
                frame = read_buffer.get()
                if frame is None:
                    break
                one_second_frames.append(frame)
                original_total_frame_count += 1
                continue
            if args.debug: print(f"\none_second_frames count: {len(one_second_frames)}")

            # -- calculate similarity
            if args.debug: print("=> calculate similarity")
            selected_frames = one_second_frames[:1]
            input_images = [self._get_padding_image(one_second_frames[0], padding)]
            one_second_frame_index = 0
            similarities = list()
            while one_second_frame_index < len(one_second_frames) - 1:
                temp_similarity = 1.
                I0 = input_images[-1]
                # remove duplicated frames
                one_second_frame_index += 1
                I1 = self._get_padding_image(one_second_frames[one_second_frame_index], padding)
                I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
                I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
                temp_similarity = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
                similarities.append(temp_similarity.cpu().item())
                input_images.append(I1)
                selected_frames.append(one_second_frames[one_second_frame_index])
            if len(input_images) == 1:
                input_images.append(self._get_padding_image(one_second_frames[-1], padding))
                selected_frames.append(one_second_frames[-1])
            if args.debug:
                print(f"selected frames count: {len(selected_frames)}")
                print(f"input images count: {len(input_images)}")
                print(f"similarities count: {len(similarities)}")

            # -- interpolation
            if args.debug: print("=> interpolation")
            added_frame_count = round(fps) - len(selected_frames)
            weights = np.array(similarities) / np.array(similarities).sum()
            insert_count_per_indexs = np.floor(weights * added_frame_count).astype(int)
            remaining = added_frame_count - insert_count_per_indexs.sum()
            sorted_indices = np.argsort(weights)
            for i in range(remaining):
                insert_count_per_indexs[sorted_indices[i]] += 1
            
            input_image_index = 1
            I0 = input_images[0]
            while input_image_index < len(input_images):
                I1 = input_images[input_image_index]
                add_count = insert_count_per_indexs[input_image_index - 1]
                if add_count > 0:
                    output = self.make_inference(I0, I1, add_count, scale)
                else:
                    output = []
                current_interpolated_frames = [(selected_frames[input_image_index - 1], -1)]
                for mid in output:
                    mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
                    current_interpolated_frames.append((mid[:h, :w], similarities[input_image_index - 1]))
                temp_interpolated_frames.append(current_interpolated_frames)
                input_image_index += 1
                I0 = I1
            if args.debug: print(f"temp_interpolated_frames count: {len(temp_interpolated_frames)}")

            # -- filtering interpolated frames
            if args.debug: print("=> filtering interpolated frames")
            current_total_frame_count = 0
            for row in temp_interpolated_frames:
                current_total_frame_count += len(row)
            removed_frame_count = current_total_frame_count - round(fps) + 1
            if self.debug:
                print(f"current interpolated frames: {current_total_frame_count}")
                print(f"removed interpolated {removed_frame_count} frames!")

            for row_idx, row in enumerate(temp_interpolated_frames):
                for col_idx, (frame, ssim) in enumerate(row):
                    # frame = cv2.putText(np.ascontiguousarray(frame), str(ssim), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2, 2)
                    if self.save_img:
                        cv2.imwrite(f"{self.img_dir}/{self.idx}{'' if ssim == -1 else '_interpolated'}.png", np.ascontiguousarray(frame))
                        self.idx += 1
                    write_buffer.put(np.ascontiguousarray(frame))
                    if self.debug:
                        total_interpolated_fps += 1
            temp_interpolated_frames.clear()
        
            pbar.update(original_fps)
            one_second_frames = one_second_frames[-1:]

        # -- add remained interplated frames
        if temp_interpolated_frames:
            if self.debug:
                print(f"\ncurrent remained interpolated frames: {len(temp_interpolated_frames)}")
            for row_idx, row in enumerate(temp_interpolated_frames):
                for col_idx, (frame, ssim) in enumerate(row):
                    if self.save_img:
                        cv2.imwrite(f"{self.img_dir}/{self.idx}{'' if ssim == -1 else '_interpolated'}.png", np.ascontiguousarray(frame))
                        self.idx += 1
                    write_buffer.put(np.ascontiguousarray(frame))
                    if self.debug:
                        total_interpolated_fps += 1
            temp_interpolated_frames.clear()

        write_buffer.put(None)

        if self.debug:
            print(f"\ntotal interpolated frames: {total_interpolated_fps}")
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
        

if __name__ == '__main__':
    
    r"""
    Example Command Line:

        > python inference_v3.py --video=C:\Users\gemiso\Desktop\frame_interpol_videos\\son_goal_01_25fps.mp4 --fps=29.97 --bitrate=60M
        > python inference_v3.py --video=C:\Users\gemiso\Desktop\frame_interpol_videos\\son_goal_01_25fps.mp4 --fps=59.94 --bitrate=60M
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
    rife.run(args.video, args.ext, args.fps, args.bitrate, scale=0.5)
