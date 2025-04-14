
recording_format = 'csv'  # csv/ bvh/ ...
controller_type = 'plane'   # fight/ plane/ terrain / ...
animation_type = 'move'  # move/ action/ ...


# (anim name, start frame, end frame, 'rooting_method', add_toe, action)
settings_animations = [
    ('animations/csv/push_and_stumble1_FBX-Unity.csv', 194, 351, 'root_simple:Hips/Spine2', False, False),
]

settings_run_components = [
    # 'parse_bvh',
    'generate_database',
    # 'generate_features',
    # 'train_decompressor',
    # 'train_stepper_projector',  # in parallel
    # 'train_stepper',  # individually
    # 'train_projector',  # individually
]





