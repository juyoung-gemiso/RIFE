import os
import av
import time
import math
import torch
import warnings
import argparse
import numpy as np
import multiprocessing as mp
from utils import *
from ffmpeg_utils import *
from interpolate_options import InterpolateOptions

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
    
    def _update_flag_and_exp(self, interpolate_options:InterpolateOptions):
        if interpolate_options.deleted_one_frame_or_not:
            if interpolate_options.delete_frame_flag:
                interpolate_options.exp -= 1
            else:
                interpolate_options.exp += 1
            interpolate_options.delete_frame_flag = not interpolate_options.delete_frame_flag

    def interpolate(self, one_second_frames, video_container, video_stream, interpolate_options:InterpolateOptions):
        h, w = interpolate_options.size
        interpolated_frame_count = 0
        original_frame_start_index = 0
        for inserted_index in range(interpolate_options.interval, len(one_second_frames), interpolate_options.interval):
            left_frame = self._get_padding_image(one_second_frames[inserted_index - 1], interpolate_options.padding)
            right_frame = self._get_padding_image(one_second_frames[inserted_index], interpolate_options.padding)
            output = self.make_inference(left_frame, right_frame, interpolate_options.exp, interpolate_options.scale)
            self._update_flag_and_exp(interpolate_options)

            # insert original frames
            for original_frame_index in range(original_frame_start_index, inserted_index):
                img = one_second_frames[original_frame_index][:, :, ::-1]
                if interpolate_options.save_img:
                    save_image(img, f"{interpolate_options.output_dir}/{self.idx:06d}.tga")
                    self.idx += 1
                write_frame_with_av(img, video_container, video_stream, interpolate_options.size)
                interpolate_options.interpolated_total_frame_count += 1
            original_frame_start_index = inserted_index

            # insert interpolated frames
            for mid in output:
                mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
                mid = np.ascontiguousarray(mid[:h, :w])[:, :, ::-1]
                if interpolate_options.save_img:
                    save_image(mid, f"{interpolate_options.output_dir}/{self.idx:06d}_interpolated.tga")
                    self.idx += 1
                write_frame_with_av(img, video_container, video_stream, interpolate_options.size)
                interpolate_options.interpolated_total_frame_count += 1
                if self.debug: interpolated_frame_count += 1
    
        if self.debug: print(f"interpolated_frames count: {interpolated_frame_count}")

        return one_second_frames[inserted_index:]

    def run(self, video:str, extension:str, output_fps:float, scale:float=1.0, save_img:bool=False, temp_dir:str=None):
        if temp_dir is not None:
            self.output_base_path = temp_dir
        if self.debug:
            start_time = time.time()

        # -- get metadata
        videoCapture = cv2.VideoCapture(video)
        input_fps = videoCapture.get(cv2.CAP_PROP_FPS) # 원본 영상의 fps
        total_frames = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)) * 2 # 원본 영상의 total_frames x 2 (e.g. 25i -> 50p)
        input_fps_2x = input_fps * 2
        videoCapture.release()
        container = av.open(video)
        audio_stream_count = sum(1 for stream in container.streams if stream.type == 'audio') # 원본 영상의 오디오 스트림 개수
        lastframe = next(container.decode(video=0)) # 원본 영상의 첫 번째 프레임
        # 인터레이스 영상일 경우 프레임 추출 수행
        if lastframe.interlaced_frame:
            # -- extract progressive frames from interlaced video using ffmpeg
            frames_dir = get_output_frames_path(video, self.output_base_path)
            ffmpeg_process = mp.Process(target=interlaced_to_progressive_2x_frames, args=(video, frames_dir, self.debug,))
            ffmpeg_process.start()
        else:
            print("=> not supported progressvie video yet.")
            return
        if self.debug:
            end_time = time.time()
            print(f"=> extract progressive frames time: {end_time - start_time}s")
            start_time = time.time()

        # -- calculate inserted interval
        interval = max(1, int(input_fps_2x // (round(output_fps * 2) - round(input_fps_2x))))
        exp = math.floor((output_fps * 2) / input_fps_2x)
        deleted_one_frame_or_not = False # e.g. 1, 1, 1, 1, ...
        if (output_fps * 2) % input_fps_2x != 0 and exp > 1:
            deleted_one_frame_or_not = True # e.g. 2, 1, 2, 1, ...
        if self.debug: print(f"original fps: {input_fps_2x}, total frames: {total_frames}")

        # -- get frame size
        lastframe = None
        while lastframe is None:
            if os.listdir(frames_dir):
                lastframe = read_image(os.path.join(frames_dir, os.listdir(frames_dir)[0]))
                break
        h, w, _ = lastframe.shape
        assert int((output_fps * 2) // input_fps_2x) > 0, 'You can\'t drop frames!! You must be input to fps greater than original fps!'

        # -- calculate padding
        tmp = max(32, int(32 / scale))
        ph = ((h - 1) // tmp + 1) * tmp
        pw = ((w - 1) // tmp + 1) * tmp
        if self.debug: print(f"Padding Height: {ph}, Padding Width: {pw}")
        padding = (0, pw - w, 0, ph - h)

        # -- get video writer
        output_video_name, video_container, video_stream = get_video_writer_using_av(video, round(output_fps) * 2, (h, w), extension)

        # -- interpolate extracted progressive frames using RIFE
        original_total_frame_count = 0
        interpolate_options = InterpolateOptions(
            interval=interval, 
            padding=padding, 
            exp=exp,
            scale=scale, 
            deleted_one_frame_or_not=deleted_one_frame_or_not, 
            delete_frame_flag=False, 
            output_dir='', 
            input_fps_2x=input_fps_2x, 
            interpolated_total_frame_count=0,
            size=(h, w),
            save_img=save_img
        )

        # -- extracted frame start index(end index = start index + input_fps_2x + 1)
        start_index = 0
        end_index = int(start_index + round(input_fps_2x))

        while True:
            if (os.path.exists(os.path.join(frames_dir, f"{start_index:06d}.tga")) \
                and os.path.exists(os.path.join(frames_dir, f"{end_index:06d}.tga"))) \
                    or total_frames - start_index <= round(input_fps_2x):

                    if self.debug: print(f"{start_index:06d}.tga ~ {end_index:06d}.tga")
                    # 남은 프레임이 fps보다 적을 경우
                    if total_frames - start_index <= round(input_fps_2x):
                        end_index = total_frames - 1
                        ffmpeg_process.join()
                    
                    one_second_frames = []
                    for i in range(start_index, end_index + 1):
                        one_second_frames.append(read_image(os.path.join(frames_dir, f"{i:06d}.tga")))
                        original_total_frame_count += 1

                    if self.debug: 
                        print(f"\none_second_frames count: {len(one_second_frames)}")
                        print("=> interpolation")

                    one_second_frames = self.interpolate(
                        one_second_frames,
                        video_container,
                        video_stream,
                        interpolate_options
                    )

                    # 보간 처리한 프레임 삭제
                    remove_images(frames_dir, start_index, end_index)

                    if end_index == total_frames - 1: break
                    start_index = end_index
                    end_index = int(start_index + round(input_fps_2x))

        if one_second_frames:
            for frame in one_second_frames:
                write_frame_with_av(frame[:, :, ::-1], video_container, video_stream, interpolate_options.size)
                interpolate_options.interpolated_total_frame_count += 1
                
        if self.debug:
            print(f"\ninterpolated_total_frame_count: {interpolate_options.interpolated_total_frame_count}")
            print(f"original_total_frame_count: {original_total_frame_count}")

        if self.debug:
            end_time = time.time()
            print(f"=> interpolation time: {end_time - start_time}s")
            start_time = time.time()

        if video_container is not None:
            for packet in video_stream.encode():
                video_container.mux(packet)
            video_container.close()
            
        output_path = transferAudio(video, output_video_name, output_fps, extension, audio_stream_count)

        if self.debug:
            end_time = time.time()
            print(f"=> transferAudio time: {end_time - start_time}s")
        self.idx = 0

        shutil.rmtree(frames_dir)

        return output_path

if __name__ == '__main__':
    r"""
    Example Command Line:

        > python inference_v7.py --video=Z:\\AI_workspace\\video_interpolation_backup\\soccer_25fps_1m30s.mxf --output_base_path=Z:\\AI_workspace\\video_interpolation_backup --fps=29.97 --ext=mxf
    """
    parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
    parser.add_argument('--video', dest='video', type=str, default=None)
    parser.add_argument('--output_base_path', dest='output_base_path', type=str, default=None)
    parser.add_argument('--fps', dest='fps', type=float, default=None)
    parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='vid_out video extension')
    parser.add_argument('--save_img', dest='save_img', action='store_true', help='save image or not')
    parser.add_argument('--debug', dest='debug', action='store_true', help='whether debug or not')
    args = parser.parse_args()
    
    rife = RIFE(model_dir="train_log", output_base_path=args.output_base_path, debug=args.debug)
    rife.run(args.video, args.ext, args.fps, scale=0.25, save_img=args.save_img)
