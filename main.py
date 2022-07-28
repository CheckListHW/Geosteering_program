from Geosteering_model import GeosteeringModel

if __name__ == '__main__':
    print('start')
    xml_path = 'input/Offset_1_BAJ.xml'
    GR_path = 'input/GR_Well_6.las'

    model = GeosteeringModel(xml_path=xml_path, GR_path=GR_path)

    model.init_algorithm_params(5, 30, 1, 'lp_distance')
    model.start_geosteering(plot_matching=True)
    print('finish')
