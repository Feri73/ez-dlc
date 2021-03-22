import importlib

name = 'default'
experimenter = 'name'
frame_config = importlib.import_module('configurations.frames.default')
marker_config = importlib.import_module('configurations.markers.default')
data = ['label_name1', 'label_name2']
network_type = 'resnet50'
