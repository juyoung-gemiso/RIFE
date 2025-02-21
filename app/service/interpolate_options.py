from dataclasses import dataclass

@dataclass
class InterpolateOptions:
    interval: int
    padding: tuple[int, int, int]
    exp: int
    scale: float
    deleted_one_frame_or_not: bool
    delete_frame_flag: bool
    output_dir: str
    input_fps_2x: float
    interpolated_total_frame_count: int
    # (h, w)
    size: tuple[int, int]
    save_img: bool
