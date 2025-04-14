
recording_format = 'csv'  # csv/ bvh/ ...
controller_type = 'plane'   # fight/ plane/ terrain / ...
animation_type = 'move'  # move/ action/ ...


# (anim name, start frame, end frame, 'rooting_method', add_toe, action)
settings_animations = [
    ('animations/csv/plane/move/push-and-stumble_ubisoft.csv', 190, 350, 'root_simple:Hips/Spine2', False, False),
    ('animations/csv/plane/move/walk1_ubisoft.csv', 80, 6000, 'root_simple:Hips/Spine2', False, False),
    ('animations/csv/plane/move/run1_ubisoft.csv', 90, 7050, 'root_simple:Hips/Spine2', False, False),
]

settings_run_components = [
    # 'parse_bvh',
    # 'generate_database',
    'generate_features',
    'train_decompressor',
    # 'train_stepper_projector',  # in parallel
    # 'train_stepper',  # individually
    # 'train_projector',  # individually
]





