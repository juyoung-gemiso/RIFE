import os
import cv2
import time
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
    def __init__(self, model_dir, debug=False):
        self.debug = debug
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        self.model = load_model(model_dir)

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
        exp = int(fps // original_fps)
        if fps % original_fps == 0:
            exp -= 1
        if exp == 0:
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

        I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.
        I1 = pad_image(I1, padding)
        temp = None # save lastframe when processing static frame

        temp_interpolated_frames = []
        total_interpolated_fps = 0

        while True:
            if temp is not None:
                frame = temp
                temp = None
            else:
                frame = read_buffer.get()
            if frame is None:
                break
            I0 = I1
            I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.
            I1 = pad_image(I1, padding)
            I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
            I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
            ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

            break_flag = False
            skip_flag = False
            # 프레임끼리 완전히 다를때 그대로 사용
            # if ssim <= 0:
            #     temp_interpolated_frames.append([(lastframe, -1)] + [((((I0[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0))), 1) for _ in range(exp)])
            #     pbar.update(1)
            #     lastframe = frame
            #     skip_flag = True
            # 유사도가 높은(중복된) 프레임의 경우 다음 프레임을 사용하여 interpolation한 프레임을 I1으로 사용
            # elif ssim > 0.996:
            #     frame = read_buffer.get() # read a new frame
            #     if frame is None:
            #         break_flag = True
            #         frame = lastframe
            #     else:
            #         temp = frame
            #     I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.
            #     I1 = pad_image(I1, padding)
            #     I1 = self.model.inference(I0, I1, scale)
            #     I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
            #     ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
            #     frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
            #     # I0와 새로운 I1의 유사도가 낮을 경우 I0만 사용
            #     if ssim < 0.2:
            #         temp_interpolated_frames.append([(lastframe, -1)] + [((((I0[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0))), 1) for _ in range(exp)])
            #         pbar.update(1)
            #         lastframe = frame
            #         skip_flag = True
            
            if not skip_flag:
                output = self.make_inference(I0, I1, exp, scale)
                
                current_interplated_frames = []
                current_interplated_frames.append((lastframe, -1))
                for mid in output:
                    mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
                    current_interplated_frames.append((mid[:h, :w], ssim.cpu().item() if not isinstance(ssim, float) else ssim))
                temp_interpolated_frames.append(current_interplated_frames)

            if len(temp_interpolated_frames) >= round(original_fps):
                current_total_frame_count = 0
                for row in temp_interpolated_frames:
                    current_total_frame_count += len(row)
                removed_frame_count = current_total_frame_count - round(fps)
                if self.debug:
                    print(f"\ncurrent interpolated frames: {current_total_frame_count}")
                    print(f"removed interpolated {removed_frame_count} frames!!")

                # 첫 iteration에는 가장 큰 유사도를 가지는 frame을 삭제하고, 그 다음부터 그 다음 크기의 유사도를 가지는 frame을 삭제
                target_frame_index = []
                nth_largest = 0
                while removed_frame_count > 0:
                    for row_idx, row in enumerate(temp_interpolated_frames):
                        col_idx = sorted(list(range(len(row))), key=lambda i: row[i][1], reverse=True)[nth_largest]
                        if row[col_idx][1] == -1:
                            continue
                        target_frame_index.append((row[col_idx][1], row_idx, col_idx)) # ssim, row, col
                        removed_frame_count -= 1
                        if removed_frame_count == 0:
                            break
                    nth_largest += 1

                for row_idx, row in enumerate(temp_interpolated_frames):
                    for col_idx, (frame, ssim) in enumerate(row):
                        if (ssim, row_idx, col_idx) not in target_frame_index:
                            frame = cv2.putText(np.ascontiguousarray(frame), str(ssim), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2, 2)
                            write_buffer.put(frame)
                if self.debug:
                    total_interpolated_fps += round(fps)
                temp_interpolated_frames.clear()
            
            pbar.update(1)
            lastframe = frame
            if break_flag:
                break

        if temp_interpolated_frames:
            if self.debug:
                print(f"\ncurrent remained interpolated frames: {len(temp_interpolated_frames)}")
            total_interpolated_fps += len(temp_interpolated_frames)
            for row_idx, row in enumerate(temp_interpolated_frames):
                for col_idx, (frame, ssim) in enumerate(row):
                    write_buffer.put(frame)
            temp_interpolated_frames.clear()

        write_buffer.put(lastframe)
        write_buffer.put(None)

        if self.debug:
            print(f"\ntotal interpolated frames: {total_interpolated_fps + 1}")

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
    """
    Example Command Line:

        > python inference.py --video=E:\\video_enhance\\frame_interpol\\son_goal_01_25fps.mp4 --fps=29.97 --bitrate=60M
        > python inference.py --video=E:\\video_enhance\\frame_interpol\\son_goal_01_25fps.mp4 --fps=59.94 --bitrate=60M
    """
    parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
    parser.add_argument('--video', dest='video', type=str, default=None)
    parser.add_argument('--output', dest='output', type=str, default=None)
    parser.add_argument('--fps', dest='fps', type=float, default=None)
    parser.add_argument('--bitrate', dest='bitrate', type=str, default=None, help="e.g. 60M")
    parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='vid_out video extension')
    parser.add_argument('--debug', dest='debug', action='store_true', help='whether debug or not')
    args = parser.parse_args()
    
    rife = RIFE(model_dir="train_log", debug=args.debug)
    rife.run(args.video, args.ext, args.fps, args.bitrate, scale=1.0)
