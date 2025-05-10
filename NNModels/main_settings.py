
recording_format = 'csv'  # csv/ bvh/ ...
controller_type = 'fight'   # fight/ plane/ terrain / ...
animation_type = 'actions'  # move/ action/ none/ ...


# (anim name, start frame, end frame, 'rooting_method', add_toe, action_params, speed_up_factor)
# actions: the animations folder's names must be action tags
settings_animations = [
    ('animations/{0}/{1}/{2}/1/punches_mixamo_1-1.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (0, 10), .57),
    ('animations/{0}/{1}/{2}/1/punches_mixamo_1-2.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (0, 20), .57),
    ('animations/{0}/{1}/{2}/1/punches_mixamo_1-3.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (0, 10), .57),
    ('animations/{0}/{1}/{2}/1/punches_mixamo_1-4.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (0, 0), .57),
    ('animations/{0}/{1}/{2}/1/punches_mixamo_1-5.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (0, 0), .57),
    ('animations/{0}/{1}/{2}/1/punches_mixamo_1-6.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (0, 0), .57),
    ('animations/{0}/{1}/{2}/1/punches_mixamo_1-7.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (0, 0), .57),
    ('animations/{0}/{1}/{2}/1/punches_mixamo_1-8.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (0, 0), .57),
    ('animations/{0}/{1}/{2}/1/punches_mixamo_1-9.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (0, 0), .57),
    ('animations/{0}/{1}/{2}/1/punches_mixamo_1-10.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (0, 0), .57),
    ('animations/{0}/{1}/{2}/1/punches_mixamo_1-11.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (0, 0), .57),
    ('animations/{0}/{1}/{2}/1/punches_mixamo_1-12.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (0, 0), .57),
    ('animations/{0}/{1}/{2}/1/punches_mixamo_1-14.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (25, 0), .57),
    ('animations/{0}/{1}/{2}/1/punches_mixamo_1-15.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (35, 0), .57),
    ('animations/{0}/{1}/{2}/1/punches_mixamo_1-16.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (0, 0), .57),
    ('animations/{0}/{1}/{2}/1/punches_mixamo_1-17.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (10, 0), .57),
    ('animations/{0}/{1}/{2}/2/punches_mixamo_2-2.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (0, 12), .57),
    ('animations/{0}/{1}/{2}/2/punches_mixamo_2-3.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (2, 10), .57),
    ('animations/{0}/{1}/{2}/2/punches_mixamo_2-4.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (0, 0), .57),
    ('animations/{0}/{1}/{2}/2/punches_mixamo_2-5.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (0, 0), .57),
    ('animations/{0}/{1}/{2}/2/punches_mixamo_2-6.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (0, 5), .57),
    ('animations/{0}/{1}/{2}/2/punches_mixamo_2-7.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (3, 10), .57),
    ('animations/{0}/{1}/{2}/2/punches_mixamo_2-8.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (3, 30), .57),
    ('animations/{0}/{1}/{2}/2/punches_mixamo_2-9.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (2, 7), .57),
    ('animations/{0}/{1}/{2}/2/punches_mixamo_2-10.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (2, 0), .57),
    ('animations/{0}/{1}/{2}/2/punches_mixamo_2-11.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (3, 18), .57),
    ('animations/{0}/{1}/{2}/2/punches_mixamo_2-12.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (5, 8), .57),
    ('animations/{0}/{1}/{2}/2/punches_mixamo_2-13.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (5, 20), .57),
    ('animations/{0}/{1}/{2}/2/punches_mixamo_2-14.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (5, 20), .57),
    ('animations/{0}/{1}/{2}/2/punches_mixamo_2-15.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (0, 40), .57),
    ('animations/{0}/{1}/{2}/2/punches_mixamo_2-16.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (0, 20), .57),
    ('animations/{0}/{1}/{2}/2/punches_mixamo_2-17.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (0, 20), .57),
    ('animations/{0}/{1}/{2}/3/punches_mixamo_3-1.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (0, 10), .57),
    ('animations/{0}/{1}/{2}/3/punches_mixamo_3-2.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (0, 36), .57),
    ('animations/{0}/{1}/{2}/3/punches_mixamo_3-3.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (12, 4), .57),
    ('animations/{0}/{1}/{2}/3/punches_mixamo_3-4.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (0, 6), .57),
    ('animations/{0}/{1}/{2}/3/punches_mixamo_3-5.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (10, 27), .57),
    ('animations/{0}/{1}/{2}/3/punches_mixamo_3-6.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (0, 24), .57),
    ('animations/{0}/{1}/{2}/3/punches_mixamo_3-7.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (0, 10), .57),
    ('animations/{0}/{1}/{2}/3/punches_mixamo_3-8.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (0, 50), .57),
    ('animations/{0}/{1}/{2}/3/punches_mixamo_3-9.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (0, 14), .57),
    ('animations/{0}/{1}/{2}/3/punches_mixamo_3-10.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (0, 18), .57),
    ('animations/{0}/{1}/{2}/3/punches_mixamo_3-11.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (35, 15), .57),
    ('animations/{0}/{1}/{2}/3/punches_mixamo_3-12.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (25, 35), .57),
    ('animations/{0}/{1}/{2}/4/punches_mixamo_4-1.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (20, 15), .57),
    ('animations/{0}/{1}/{2}/4/punches_mixamo_4-2.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (0, 10), .57),
    ('animations/{0}/{1}/{2}/4/punches_mixamo_4-3.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (0, 5), .57),
    ('animations/{0}/{1}/{2}/4/punches_mixamo_4-4.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (0, 33), .57),
    ('animations/{0}/{1}/{2}/4/punches_mixamo_4-5.{0}'
     .format(recording_format, controller_type, animation_type), 1, -1, 'root_smoothed:Hips/Hips:cma:15', False, (24, 0), .57),
]

settings_run_components = [
    # 'parse_bvh',
    # 'generate_database',
    # 'generate_features',
    # 'train_decompressor',
    # 'train_stepper_projector',  # in parallel
    # 'train_stepper',  # individually
    'train_projector',  # individually
]





