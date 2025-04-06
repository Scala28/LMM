
settings_type = 'terrain'   # fight/ plane/ terrain / ...
settings_animation_type = 'move'  # move/ action/ ...

# (anim name, start frame, end frame, 'rooting_method', add_toe, action)
settings_animations = [
    ('raw_animations/terrain/move/push-and-stumble_ubisoft.bvh', 194, 351, 'root_simple:Hips/Spine2', False, False),
    ('raw_animations/terrain/move/run1_ubisoft.bvh', 300, 6000, 'root_simple:Hips/Spine2', False, False),
    ('raw_animations/terrain/move/walk1_ubisoft.bvh', 200, 6000, 'root_simple:Hips/Spine2', False, False),
    ('raw_animations/terrain/move/obstacles1_ubisoft.bvh', 80, 6000, 'root_simple:Hips/Spine2', False, False),
    ('raw_animations/terrain/move/obstacles2_ubisoft.bvh', 300, 7000, 'root_simple:Hips/Spine2', False, False),
    ('raw_animations/terrain/move/obstacles6_ubisoft.bvh', 100, 7000, 'root_simple:Hips/Spine2', False, False),
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





