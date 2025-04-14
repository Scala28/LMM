import subprocess
import my_modules.parser_mixamo as parser
import shutil
from datetime import datetime
import os
import main_settings as settings


def parse_input_animations(animations_list):
    pth = 'animations/{0}/{1}/'.format(settings.controller_type, settings.animation_type)
    for anim in animations_list:
        anim_name = anim[0].split('/')[-1]

        print("processing mixamo format ...")
        parser.process_bvh(anim[0], pth + anim_name)


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

    if settings.animation_type == 'move':
        if 'generate_database' in settings.settings_run_components:
            print("Generating database ...")
            subprocess.run([venv_path, "generate/generate_database_{0}.py".format(settings.controller_type)])
            print("Generating root direction bvh ...")
            subprocess.run([venv_path, "generate/root_direction_tester.py".format(settings.controller_type)])

        if 'generate_features' in settings.settings_run_components:
            print("Generating features ...")
            subprocess.run([venv_path, "generate/generate_features_{0}.py".format(settings.controller_type)])

        if 'train_decompressor' in settings.settings_run_components:
            print("Starting decompressor training ...")
            subprocess.run([venv_path, "trainings/train_decompressor_{0}.py".format(settings.controller_type)])

        if 'train_projector' in settings.settings_run_components:
            print("Starting projector training ...")
            subprocess.run([venv_path, "trainings/train_projector.py"])

        if 'train_stepper' in settings.settings_run_components:
            print("Starting stepper training ...")
            subprocess.run([venv_path, "trainings/train_stepper.py"])

        if 'train_stepper_projector' in settings.settings_run_components:
            print("Starting stepper and projector training ...")
            process1 = subprocess.Popen([venv_path, "trainings/train_stepper.py"])
            process2 = subprocess.Popen([venv_path, "trainings/train_projector.py"])
            process1.wait()
            process2.wait()


