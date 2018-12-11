if __name__ == '__main__':
    help()
    low_dpi = False
    lagrange_points = [1, 2]
    orbit_types = ['horizontal']
    c_levels = [3.05, 3.1, 3.15]

    orbit_ids = {'horizontal':  {1: {3.05: 808, 3.1: 577, 3.15: 330}, 2: {3.05: 1066, 3.1: 760, 3.15: 373}},
                 'halo':  {1: {3.05: 1235, 3.1: 836, 3.15: 358}, 2: {3.05: 1093, 3.1: 651, 3.15: 0}},
                 'vertical': {1: {3.05: 1664, 3.1: 1159, 3.15: 600}, 2: {3.05: 1878, 3.1: 1275, 3.15: 513}}}

    for orbit_type in orbit_types:
        for lagrange_point in lagrange_points:
            for c_level in c_levels:
                display_augmented_validation = DisplayAugmentedValidation(orbit_type, lagrange_point,
                                                                              orbit_ids[orbit_type][lagrange_point][
                                                                                  c_level],
                                                                              low_dpi=low_dpi)

                del display_augmented_validation