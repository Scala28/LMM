
recording_format = 'csv'  # csv/ bvh/ ...
controller_type = 'plane'   # fight/ plane/ terrain / ...
animation_type = 'move'  # move/ action/ ...


# (anim name, start frame, end frame, 'rooting_method', add_toe, action)

"""settings_animations = [
    ('animations/{0}/{1}/{2}/push-and-stumble_ubisoft.{0}'.
     format(recording_format, controller_type, animation_type), 190, 350, 'root_simple:Hips/Spine2', False, False),
    ('animations/{0}/{1}/{2}/walk1_ubisoft.{0}'.
     format(recording_format, controller_type, animation_type), 80, 6000, 'root_simple:Hips/Spine2', False, False),
    ('animations/{0}/{1}/{2}/run1_ubisoft.{0}'.
     format(recording_format, controller_type, animation_type), 90, 7050, 'root_simple:Hips/Spine2', False, False),
    ('animations/{0}/{1}/{2}/obstacles1_ubisoft.{0}'.
     format(recording_format, controller_type, animation_type), 200, 4500, 'root_simple:Hips/Spine2', False, False),
    ('animations/{0}/{1}/{2}/obstacles2_ubisoft.{0}'.
     format(recording_format, controller_type, animation_type), 160, 5700, 'root_simple:Hips/Spine2', False, False),
    ('animations/{0}/{1}/{2}/obstacles6_ubisoft.{0}'.
     format(recording_format, controller_type, animation_type), 120, 6600, 'root_simple:Hips/Spine2', False, False),
]"""
settings_animations = [
    ('animations/{0}/{1}/{2}/push-and-stumble_ubisoft.{0}'.
     format(recording_format, controller_type, animation_type), 190, 350, 'root_simple:Hips/Spine2', False, False),
    ('animations/{0}/{1}/{2}/walk1_ubisoft.{0}'.
     format(recording_format, controller_type, animation_type), 80, 6000, 'root_simple:Hips/Spine2', False, False),
    ('animations/{0}/{1}/{2}/run1_ubisoft.{0}'.
     format(recording_format, controller_type, animation_type), 90, 7050, 'root_simple:Hips/Spine2', False, False),
]

settings_run_components = [
    # 'parse_bvh',
    # 'generate_database',
    'generate_features',
    'train_decompressor',
    'train_stepper_projector',  # in parallel
    # 'train_stepper',  # individually
    # 'train_projector',  # individually
]





