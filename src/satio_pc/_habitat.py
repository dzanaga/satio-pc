
RSI_META_S2_HABITAT = {
    'ndvi': {'bands': ['B08', 'B04'],
             'range': [-1, 1]},

    'ndmi': {'bands': ['B08', 'B11'],
             'range': [-1, 1]},

    'nbr': {'bands': ['B08', 'B12'],
            'range': [-1, 1]},

    'nbr2': {'bands': ['B11', 'B12'],
             'range': [-1, 1]},

    'evi': {'bands': ['B8A', 'B04', 'B02'],
            'range': [-3, 3]},

    'savi': {'bands': ['B08', 'B04'],
             'range': [-1, 1]},

    'hsvh': {'bands': ['B04', 'B03', 'B02'],
             'range': [0, 1]},

    'hsvv': {'bands': ['B04', 'B03', 'B02'],
             'range': [0, 1]},

    'rep': {'bands': ['B04', 'B07', 'B05', 'B06'],
            'range': [500, 900]},

    'anir': {'bands': ['B04', 'B08', 'B11'],
             'range': [0, 1]},

    'nirv': {'bands': ['B08', 'B04'],
             'range': [0, 1]},

    'auc': {'bands': ['B02', 'B03', 'B04',
                      'B08', 'B11', 'B12'],
            'range': [0, 1]},

    'nauc': {'bands': ['B02', 'B03', 'B04',
                       'B08', 'B11', 'B12'],
             'range': [0, 1]},

    # ndwi (GAO, 1996)
    'ndwi': {'bands': ['B8A', 'B12'],
             'range': [-1, 1]},

    # ndwi (McFeeters, 1996)
    'ndwi2': {'bands': ['B03', 'B8A'],
              'range': [-1, 1],
              'func': 'norm_diff'},

    # modified NDWI (Xu, 2006)
    'mndwi': {'bands': ['B03', 'B11'],
              'range': [-1, 1]},

    # normalized difference greenness index
    'ndgi': {'bands': ['B03', 'B04'],
             'range': [-1, 1]},

    # advanced Vegetation Index (AVI)
    'avi': {'bands': ['B04', 'B08'],
            'range': [0, 1]},

    # bare soil index
    'bsi': {'bands': ['B02', 'B04', 'B08', 'B11'],
            'range': [-1, 1]},

    # brightness (as defined in sen2agri)
    'brightness': {'bands': ['B03', 'B04', 'B08', 'B11'],
                   'range': [0, 1]},

    # series of normalized difference red edge indices
    'ndre1': {'bands': ['B08', 'B05'],
              'range': [-1, 1]},

    'ndre2': {'bands': ['B08', 'B06'],
              'range': [-1, 1]},

    'ndre3': {'bands': ['B08', 'B07'],
              'range': [-1, 1]},

    'ndre4': {'bands': ['B06', 'B05'],
              'range': [-1, 1]},

    'ndre5': {'bands': ['B07', 'B05'],
              'range': [-1, 1]},

    # improved NDVI
    'atsavi': {'bands': ['B08', 'B04'],
               'range': [-1, 1]},

    # chlorophyll index
    'lci': {'bands': ['B08', 'B05', 'B04'],
            'range': [-10, 10]}
}
