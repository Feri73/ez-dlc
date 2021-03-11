import importlib

name = 'default'
frame_config = importlib.import_module('configurations.frames.default')
marker_config = importlib.import_module('configurations.markers.default')
data = ['label_name1', 'label_name2']
training_data_frame_count = 20
network_type = 'resnet50'
