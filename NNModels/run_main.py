import subprocess
import my_modules.parser_default_to_ubisoft as parser_def
import my_modules.parser_mixamo_to_ubisoft as parser_mix
import shutil
from datetime import datetime
import os
import main_settings as settings
import logging

logging.basicConfig(
    filename='bvh_filter.log',  # Log file name
    level=logging.INFO,  # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def parse_input_animations(animations_list):
    pth = ''
    if settings.settings_animation_type == 'move':
        pth = 'animations/{0}/move/'.format(settings.settings_type)
    elif settings.settings_animation_type == 'action':
        pth = 'animations/{0}/action/'.format(settings.settings_type)

    for i in range(len(animations_list)):
        anim_name = animations_list[i][0].split('/')[-1]
        add_toe = animations_list[i][4]

        if settings.bvh_type == 'default':
            print("processing default format ...")
            parser_def.process_bvh(animations_list[i][0], pth + anim_name)
        elif settings.bvh_type == 'mixamo':
            print("processing mixamo format ...")
            parser_mix.process_bvh(animations_list[i][0], pth + anim_name, add_toe)


def backup_old_executions_settings():
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    old_dir_list = [name for name in os.listdir('old_runs') if os.path.isdir(os.path.join('old_runs', name))]

    dir_number = 0
    for dir_name in old_dir_list:
        if dir_number < int(dir_name.split('_')[0]):
            dir_number = int(dir_name.split('_')[0])

    dir_number += 1
    dir_name = str(dir_number) + '_' + current_time.replace('-', '_').replace(':', '').replace(' ', '_')
    os.makedirs('old_runs/' + dir_name, exist_ok=True)
    shutil.copy("main_settings.py", 'old_runs/'+dir_name+'/main_settings.txt'.format(type))


if __name__ == "__main__":
    # Move the previous main_settings to old_runs
    backup_old_executions_settings()

    if 'parse_bvh' in settings.settings_run_components:
        print("Starting BVH parsing ...")
        parse_input_animations(settings.settings_animations)

    venv_path = "C:/Users/lucas/AppData/Local/Programs/Python/Python39/python.exe"

    if settings.settings_animation_type == 'move':
        if 'generate_database' in settings.settings_run_components:
            print("Generating database ...")
            subprocess.run([venv_path, "generate/generate_database_{0}.py".format(settings.settings_type)])
            print("Generating root direction bvh ...")
            subprocess.run([venv_path, "generate/root_direction_tester.py".format(settings.settings_type)])

        if 'generate_features' in settings.settings_run_components:
            print("Generating features ...")
            subprocess.run([venv_path, "generate/generate_features_{0}.py".format(settings.settings_type)])

        if 'train_decompressor' in settings.settings_run_components:
            print("Starting decompressor training ...")
            subprocess.run([venv_path, "trainings/train_decompressor_{0}.py".format(settings.settings_type)])

        if 'train_projector' in settings.settings_run_components:
            print("Starting projector training ...")
            subprocess.run([venv_path, "trainings/train_projector_{0}.py".format(settings.settings_type)])

        if 'train_stepper' in settings.settings_run_components:
            print("Starting stepper training ...")
            subprocess.run([venv_path, "trainings/train_stepper_{0}.py".format(settings.settings_type)])

        if 'train_stepper_projector' in settings.settings_run_components:
            print("Starting stepper and projector training ...")
            process1 = subprocess.Popen([venv_path, "trainings/train_stepper_{0}.py".format(settings.settings_type)])
            process2 = subprocess.Popen([venv_path, "trainings/train_projector_{0}.py".format(settings.settings_type)])
            process1.wait()
            process2.wait()


