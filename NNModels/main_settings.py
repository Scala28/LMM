
settings_type = 'fight'   # fight/ plane/ terrain / ...
settings_animation_type = 'move'  # move/ action/ ...
bvh_type = 'mixamo'  # mixamo/ default/ ...

# (anim name, start frame, end frame, 'rooting_method', add_toe, action)
settings_animations = [
    ('raw_animations/fight/move/take-1_mixamo.bvh', 1000, 13000, 'root_smoothed:Hips/Hips:cma:150', 'true', 'false'),
    ('raw_animations/fight/move/take-3_mixamo.bvh', 1500, 6000, 'root_smoothed:Hips/Hips:cma:150','true', 'false'),
    ('raw_animations/fight/move/take-5_mixamo.bvh', 200, 12000, 'root_locked:Hips:0,0,1.3', 'true', 'false'),
    ('raw_animations/fight/move/take-6_mixamo.bvh', 200, 5000, 'root_locked:Hips:0,0,1', 'true', 'false'),
    ('raw_animations/fight/move/take-7_mixamo.bvh', 200, 5000, 'root_locked:Hips:0,0,1.5', 'true','false'),
]

settings_run_components = [
    # 'parse_bvh',
    # 'generate_database',
    # 'generate_features',
    'train_decompressor',
    # 'train_stepper_projector',  # in parallel
    # 'train_stepper',  # individually
    # 'train_projector',  # individually
]





