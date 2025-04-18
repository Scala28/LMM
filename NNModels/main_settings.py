
recording_format = 'csv'  # csv/ bvh/ ...
controller_type = 'fight'   # fight/ plane/ terrain / ...
animation_type = 'move'  # move/ action/ ...


# (anim name, start frame, end frame, 'rooting_method', add_toe, action)

settings_animations = [
    ('animations/{0}/{1}/{2}/take-1_boxe.{0}'.
     format(recording_format, controller_type, animation_type), 1000, 15000, 'root_smoothed:Hips/Head:cma:150', False, False),
    ('animations/{0}/{1}/{2}/take-4_boxe.{0}'.
     format(recording_format, controller_type, animation_type), 4000, 10000, 'root_smoothed:Hips/Head:cma:150', False, False),
    ('animations/{0}/{1}/{2}/take-6.2_boxe.{0}'.
     format(recording_format, controller_type, animation_type), 200, 6000, 'root_locked:Hips:0,0,3', False, False),
    ('animations/{0}/{1}/{2}/take-6_boxe.{0}'.
     format(recording_format, controller_type, animation_type), 200, 6000, 'root_locked:Hips:0,0,3', False, False),
    ('animations/{0}/{1}/{2}/take-5_boxe.{0}'.
     format(recording_format, controller_type, animation_type), 4000, 15000, 'root_locked:Hips:0,0,3', False, False),
]
settings_run_components = [
    # 'parse_bvh',
    # 'generate_database',
    # 'generate_features',
    # 'train_decompressor',
    'train_stepper_projector',  # in parallel
    # 'train_stepper',  # individually
    # 'train_projector',  # individually
]





