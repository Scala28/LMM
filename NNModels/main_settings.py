
settings_type = 'plane'   # fight/ plane/ terrain / ...
settings_animation_type = 'move'  # move/ action/ ...

# (anim name, start frame, end frame, 'rooting_method', add_toe, action)
settings_animations = [
    ('raw_animations/plane/move/push-and-stumble_ubisoft.bvh', 194, 351, 'root_simple:Hips/Spine2', False, False),
    ('raw_animations/plane/move/run1_ubisoft.bvh', 90, 7086, 'root_simple:Hips/Spine2', False, False),
    ('raw_animations/plane/move/walk1_ubisoft.bvh', 80, 6000, 'root_simple:Hips/Spine2', False, False),
]

settings_run_components = [
    # 'parse_bvh',
    # 'generate_database',
    # 'generate_features',
    # 'train_decompressor',
    # 'train_stepper_projector',  # in parallel
    'train_stepper',  # individually
    # 'train_projector',  # individually
]





