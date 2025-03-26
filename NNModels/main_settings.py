settings_type = 'fight'   # fight/ plane/ terrain / ...
settings_animation_type = 'move'  # move/ action/ ...
bvh_type = 'mixamo'  # mixamo/ default/ ...

# settings_type = 'fight' # can be fight, plane, terrain (and futures)
# settings_animation_type = 'move' # can be move, action
# settings_animations = [
#    ('input/move/move_1_mixamo.bvh', 1000, 13000, 'root_smoothed:Hips/Hips:cma:150', 'true', 'false'),
#    ('input/move/move_3_mixamo.bvh', 1500, 6000, 'root_smoothed:Hips/Hips:cma:150','true', 'false'),
#    ('input/move/move_5_mixamo.bvh', 200, 12000, 'root_locked:Hips:0,0,1.3', 'true', 'false'),
#    ('input/move/move_6_mixamo.bvh', 200, 5000, 'root_locked:Hips:0,0,1', 'true', 'false'),
#    ('input/move/move_7_mixamo.bvh', 200, 5000, 'root_locked:Hips:0,0,1.5', 'true','false'),
# ]
# (anim name, start frame, end frame, 'rooting_method', add_toe, action)
settings_animations = [
   # ('raw_animations/fight/move/take-1_mixamo.bvh', 3000, 10000, 'root_smoothed:Hips/Head:cma:400', True, 'false'),
   ('raw_animations/fight/move/take-2_mixamo.bvh', 1500, 3000, 'root_smoothed:Hips/Head:75', True, 'false'),
   ('raw_animations/fight/move/take-5_mixamo.bvh', 1000, 6000, 'root_smoothed:Hips/Head:400', True, 'false'),
   ('raw_animations/fight/move/take-4_mixamo.bvh', 1500, 15000, 'root_smoothed:Hips/Head:400', True, 'false'),
   # ('raw_animations/fight/move/take-6_mixamo.bvh', 200, 4000, 'root_locked:Hips:0, 0, 3.3', True, 'false'),
   # ('raw_animations/fight/move/take-6_mixamo-2.bvh', 200, 4000, 'root_locked:Hips:0, 0, 3.3', True, 'false'),
   # ('raw_animations/fight/move/take-7_mixamo.bvh', 200, 6000, 'root_locked:Hips:0,0,3.3', True, 'false'),
]

settings_run_components = [
    'parse_bvh',
    'generate_database',
    'generate_features',
    # 'train_decompressor',
    # 'train_stepper_projector', # in parallel
    # 'train_stepper', # individually
    # 'train_projector', # individually
]





