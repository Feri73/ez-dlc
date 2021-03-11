import shutil

shutil.copytree('.configuration_templates', 'configurations')
shutil.move('configurations/general_configs.py', 'general_configs.py')
