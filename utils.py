import os
import shutil
import time
from datetime import datetime
from typing import Tuple, List, Optional, Callable

import cv2
import deeplabcut
import numpy as np
from PIL import Image
from tqdm import tqdm


class Profiler:
    def __enter__(self):
        self.st = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.taken_time = time.time() - self.st


class VideoManager:
    def __init__(self, vid, error_msg=None):
        self.vid = vid
        if not vid.isOpened():
            raise RuntimeError(error_msg)

    def __enter__(self):
        return self.vid

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.vid.release()


def infer_edit_frame_params(crop_offset: Tuple[int, int], crop_size: Tuple[int, int], frame_size: Tuple[int, int],
                            orig_size: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    crop_offset = crop_offset or (0, 0)
    crop_size = crop_size or orig_size
    frame_size = frame_size or crop_size
    return crop_offset, crop_size, frame_size


def edit_video(src_vid_path: str, dst_vid_path: str, crop_offset: Tuple[int, int] = None,
               crop_size: Tuple[int, int] = None, frame_size: Tuple[int, int] = None,
               frame_is_colored: bool = False, fps: int = None, time_window: Tuple[int, int] = None) -> None:
    with VideoManager(cv2.VideoCapture(src_vid_path), f'{src_vid_path} has problems.') as src_vid:
        orig_fps = int(src_vid.get(cv2.CAP_PROP_FPS))

        crop_offset, crop_size, frame_size = infer_edit_frame_params(crop_offset, crop_size, frame_size,
                                                                     (int(src_vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                                      int(src_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        time_window = time_window or (0, int(src_vid.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = fps or orig_fps

        frame_step = int(orig_fps // fps)
        assert frame_step >= 1

        with VideoManager(
                cv2.VideoWriter(dst_vid_path, cv2.VideoWriter_fourcc(*'FMP4'), fps, frame_size, frame_is_colored),
                f'Cannot create {dst_vid_path}') as dst_vid:
            for t in tqdm(range(time_window[0] * orig_fps, time_window[1] * orig_fps, frame_step)):
                src_vid.set(cv2.CAP_PROP_POS_FRAMES, t)
                _, frame = src_vid.read()
                frame = edit_frame(frame, crop_offset, crop_size, frame_size, frame_is_colored)
                dst_vid.write(frame.astype(np.uint8))


def edit_frame(frame: np.ndarray, crop_offset: Tuple[int, int], crop_size: Tuple[int, int],
               frame_size: Tuple[int, int], frame_is_colored: bool) -> np.ndarray:
    crop_offset, crop_size, frame_size = infer_edit_frame_params(crop_offset, crop_size, frame_size, frame.shape[1::-1])
    frame_is_colored = frame_is_colored or False

    if frame_is_colored and (frame.ndim == 2 or (frame.ndim == 3 and frame.shape[-1] == 1)):
        frame = np.concatenate([frame.reshape(*frame.shape[:2], 1)] * 3, axis=-1)
    if not frame_is_colored and frame.ndim == 3:
        frame = np.mean(frame, axis=-1)

    if np.all(crop_offset) != 0 or crop_size != frame.shape[1::-1]:
        frame = frame[crop_offset[1]:crop_offset[1] + crop_size[1], crop_offset[0]:crop_offset[0] + crop_size[0]]

    if frame_size != frame.shape[1::-1]:
        frame = np.array(Image.fromarray(frame).resize(size=frame_size))

    return frame


def create_dlc_project(name: str, dir: str, videos: List[str], markers: List[str],
                       custom_config: Callable[[str], Optional[str]] = None, prompt_delete: bool = True, **conf):
    videos = [os.path.abspath(vid) for vid in videos]
    custom_config = custom_config or (lambda _: None)

    config_path = os.path.abspath(f'{dir}/{name}/config.yaml')

    model_path = os.path.abspath(f'{dir}/{name}')
    orig_model_path = os.path.abspath(f'{dir}/{name}' f'-faraz-{datetime.today().strftime("%Y-%m-%d")}')

    if os.path.isdir(model_path) and (not prompt_delete or
                                      input(f'delete existing "{model_path}"? ').startswith('y')):
        shutil.rmtree(model_path)

    deeplabcut.create_new_project(name, 'faraz', videos, working_directory=dir,
                                  copy_videos=False, multianimal=False)
    os.rename(orig_model_path, model_path)

    with open(config_path, 'r') as f:
        config_content = f.readlines()

    new_config_content = []
    body_part_started = False
    for line in config_content:
        if body_part_started:
            if line.startswith('-'):
                continue
            else:
                body_part_started = False

        tmp = custom_config(line)
        if tmp is None:
            if line.startswith('bodyparts:'):
                body_part_started = True
            else:
                for key in conf:
                    if line.startswith(f'{key}:'):
                        line = f'{key}: {conf[key]}\n'
                        break

            line = line.replace(orig_model_path, model_path)
        else:
            line = tmp

        new_config_content.append(line)

        if body_part_started:
            new_config_content += [f'- {marker}\n' for marker in markers]

    with open(config_path, 'w') as f:
        f.writelines(new_config_content)
