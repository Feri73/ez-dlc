import distutils.dir_util
import glob
import os
import shutil

import cv2
import deeplabcut
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from PIL import Image
from dlclive import DLCLive
from tqdm import tqdm

import configurations.main.dlc
import configurations.main.edit_videos
import configurations.main.label_frames
import configurations.main.test_system
import general_configs
from utils import edit_video, create_dlc_project, edit_frame, VideoManager, infer_edit_frame_params


def get_dlc_config_path(model_name):
    return os.path.abspath(f'{general_configs.models_dir}/{model_name}/config.yaml')


def get_export_dir(model_name):
    exports_dir = f'{general_configs.models_dir}/{model_name}/exported-models'
    if os.path.isdir(exports_dir):
        shutil.rmtree(exports_dir)

    deeplabcut.export_model(get_dlc_config_path(model_name))
    export_dir = glob.glob(os.path.abspath(f'{exports_dir}') + '/*/')[0]

    return export_dir


def get_dlc_preds(export_dir, src_vid, crop_offset, crop_size, frame_size, frame_is_colored):
    use_dlc_preprocess = abs(crop_size[0] / frame_size[0] * frame_size[1] - crop_size[1]) < 5

    if use_dlc_preprocess:
        dlc_live = DLCLive(export_dir,
                           cropping=[crop_offset[0], crop_offset[0] + crop_size[0],
                                     crop_offset[1], crop_offset[1] + crop_size[1]],
                           resize=frame_size[0] / crop_size[0])
    else:
        dlc_live = DLCLive(export_dir)

    dlc_live.init_inference(src_vid.read()[1])

    for _ in tqdm(range(int(src_vid.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ok, frame = src_vid.read()
        if not ok:
            break

        if use_dlc_preprocess:
            poses = dlc_live.get_pose(frame.astype(np.uint8))
        else:
            processed_frame = edit_frame(frame, crop_offset, crop_size, frame_size, frame_is_colored)
            poses = dlc_live.get_pose(processed_frame.astype(np.uint8))
            poses[:, 0] = poses[:, 0] * crop_size[0] / frame_size[0] + crop_offset[0]
            poses[:, 1] = poses[:, 1] * crop_size[1] / frame_size[1] + crop_offset[1]

        yield frame, poses

    dlc_live.close()


def make_program():
    tasks = {}

    def is_task(task_num, specific_config):
        def get_func(f):
            ret_f = lambda: f(specific_config)
            ret_f.__name__ = f.__name__
            tasks[task_num] = ret_f
            return ret_f

        return get_func

    is_task.tasks = tasks
    return is_task


program_task = make_program()


@program_task(1, configurations.main.edit_videos)
def task_edit_video(configs):
    for vid in configs.videos:
        video_path = f'{general_configs.videos_dir}/{vid}.avi'
        edited_video_path = f'{general_configs.videos_dir}/{vid}_{configs.video_config.name}_' \
                            f'{configs.frame_config.name}.avi'
        edit_video(video_path, edited_video_path,
                   configs.frame_config.crop_offset,
                   configs.frame_config.crop_size,
                   configs.frame_config.frame_size,
                   configs.frame_config.frame_is_colored,
                   configs.video_config.fps,
                   configs.video_config.time_window)


@program_task(2, configurations.main.label_frames)
def task_label_frames(configs):
    for vid in configs.videos:
        vid_name = f'{vid}_{configs.marker_config.name}'
        video_path = f'{general_configs.videos_dir}/{vid_name}.avi'
        shutil.copy(f'{general_configs.videos_dir}/{vid}.avi', video_path)
        create_dlc_project('tmp', '.tmp', [video_path], configs.marker_config.markers, prompt_delete=False,
                           numframes2pick=configs.training_data_frame_count)
        cfg_path = os.path.abspath('.tmp/tmp/config.yaml')

        tmp_dir = os.path.abspath(f'.tmp/tmp/labeled-data/{vid_name}')
        dst_dir = os.path.abspath(f'{general_configs.videos_dir}/labels/{vid_name}')

        deeplabcut.extract_frames(cfg_path, mode=configs.mode, algo='kmeans', crop=False)

        if os.path.isdir(dst_dir):
            distutils.dir_util.copy_tree(dst_dir, tmp_dir)
            shutil.rmtree(dst_dir)
        deeplabcut.label_frames(cfg_path)

        shutil.move(tmp_dir, dst_dir)
        os.remove(video_path)


@program_task(3, configurations.main.dlc)
def task_create_dlc_model(configs):
    def set_video(line):
        if '.tmp/tmp.avi' in line:
            return ''
        if 'crop: ' in line:
            ref_name = os.path.abspath('.tmp/tmp.avi')
            return ''.join([f'  {ref_name.replace("tmp.avi", d)}.avi:\n    crop: 0, 1000, 0, 1000\n'
                            for d in configs.model_config.data])

    create_dlc_project(configs.model_config.name, general_configs.models_dir, ['.tmp/tmp.avi'],
                       configs.model_config.marker_config.markers, custom_config=set_video,
                       default_net_type=configs.model_config.network_type)
    shutil.rmtree(f'{general_configs.models_dir}/{configs.model_config.name}/labeled-data/tmp')

    for d_name in configs.model_config.data:
        d_path = f'{general_configs.models_dir}/{configs.model_config.name}/labeled-data/{d_name}'
        shutil.copytree(f'{general_configs.videos_dir}/labels/{d_name}', d_path)
        for img_path in glob.glob(f'{d_path}/img*.png'):
            img = np.array(Image.open(img_path))
            orig_size = img.shape[1::-1]
            img = edit_frame(img,
                             configs.model_config.frame_config.crop_offset,
                             configs.model_config.frame_config.crop_size,
                             configs.model_config.frame_config.frame_size,
                             True)
            Image.fromarray(img).save(img_path)

        crop_offset, crop_size, frame_size = infer_edit_frame_params(configs.model_config.frame_config.crop_offset,
                                                                     configs.model_config.frame_config.crop_size,
                                                                     configs.model_config.frame_config.frame_size,
                                                                     orig_size)

        dframe = pd.read_hdf(f'{d_path}/CollectedData_faraz.h5')
        for k in dframe.keys():
            if k[-1] == 'x':
                dframe[k] = (dframe[k] - crop_offset[0]) * frame_size[0] / crop_size[0]
            if k[-1] == 'y':
                dframe[k] = (dframe[k] - crop_offset[1]) * frame_size[1] / crop_size[1]
        dframe.to_csv(f'{d_path}/CollectedData_faraz.csv')
        dframe.to_hdf(f'{d_path}/CollectedData_faraz.h5', 'df_with_missing', format='table', mode='w')


@program_task(4, configurations.main.dlc)
def task_build_skeleton(configs):
    deeplabcut.check_labels(get_dlc_config_path(configs.model_config.name))
    deeplabcut.SkeletonBuilder(get_dlc_config_path(configs.model_config.name))


@program_task(5, configurations.main.dlc)
def task_create_training_data(configs):
    deeplabcut.create_training_dataset(get_dlc_config_path(configs.model_config.name))


@program_task(6, configurations.main.dlc)
def task_train_dlc(configs):
    deeplabcut.train_network(get_dlc_config_path(configs.model_config.name))


@program_task(7, configurations.main.dlc)
def task_evaluate_dlc(configs):
    deeplabcut.evaluate_network(get_dlc_config_path(configs.model_config.name), plotting=True)


@program_task(8, configurations.main.test_system)
def task_evaluate_accuracy(configs):
    export_dir = get_export_dir(configs.realtime_config.model_config.name)

    cmap = cm.get_cmap('plasma')
    markers = configs.realtime_config.model_config.marker_config.markers
    marker_colors = [np.array(cmap(i / len(markers))[:3]) * 255 for i in range(len(markers))]

    for vid_name in configs.videos:
        src_vid_name = f'{general_configs.videos_dir}/{vid_name}.avi'
        dst_vid_name = f'{general_configs.videos_dir}/{vid_name}_' \
                       f'{configs.realtime_config.model_config.name}_labeled.avi'

        with VideoManager(cv2.VideoCapture(src_vid_name), f'{src_vid_name} has problems.') as src_vid:
            orig_frame_size = (int(src_vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(src_vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

            crop_offset, crop_size, frame_size = infer_edit_frame_params(
                configs.realtime_config.model_config.frame_config.crop_offset,
                configs.realtime_config.model_config.frame_config.crop_size,
                configs.realtime_config.model_config.frame_config.frame_size,
                orig_frame_size)

            with VideoManager(cv2.VideoWriter(dst_vid_name, cv2.VideoWriter_fourcc(*'FMP4'),
                                              int(src_vid.get(cv2.CAP_PROP_FPS)), orig_frame_size, True),
                              f'Cannot create {dst_vid_name}') as dst_vid:

                for frame, poses in get_dlc_preds(export_dir, src_vid, crop_offset, crop_size, frame_size,
                                                  configs.realtime_config.model_config.frame_config.frame_is_colored):
                    for pos, color in zip(poses, marker_colors):
                        prob = np.exp(pos[2] - 1)
                        pos = pos[:2].astype(np.int)

                        size = configs.dot_size

                        bg_color = np.mean(frame[max(pos[1] - size, 0):pos[1] + size,
                                           max(pos[0] - size, 0):pos[0] + size].reshape(-1, frame.shape[-1]), axis=0)
                        frame[max(pos[1] - size, 0):pos[1] + size, max(pos[0] - size, 0):pos[0] + size] = \
                            color * prob + bg_color * (1 - prob)

                    dst_vid.write(frame.astype(np.uint8))


@program_task(9, configurations.main.test_system)
def task_evaluate_fps(configs):
    export_dir = get_export_dir(configs.realtime_config.model_config.name)

    for vid_name in configs.videos:
        src_vid_name = f'{general_configs.videos_dir}/{vid_name}.avi'

        with VideoManager(cv2.VideoCapture(src_vid_name), f'{src_vid_name} has problems.') as src_vid:
            orig_frame_size = (int(src_vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(src_vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

            crop_offset, crop_size, frame_size = infer_edit_frame_params(
                configs.realtime_config.model_config.frame_config.crop_offset,
                configs.realtime_config.model_config.frame_config.crop_size,
                configs.realtime_config.model_config.frame_config.frame_size,
                orig_frame_size)

            for _ in get_dlc_preds(export_dir, src_vid, crop_offset, crop_size, frame_size,
                                   configs.realtime_config.model_config.frame_config.frame_is_colored):
                pass


prompt = ''.join([f'{task_num}: {program_task.tasks[task_num].__name__.split("_", 1)[1].replace("_", " ")}\n'
                  for task_num in program_task.tasks]) + '0: quit\n  what to do? '
while True:
    task_num = int(input(prompt))
    if task_num == 0:
        break
    program_task.tasks[task_num]()
