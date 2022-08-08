from geosteering_tools.arg_parser import arg_parser


if __name__ == '__main__':
    my_namespace, errors_messages = arg_parser()
    if errors_messages:
        [print(er) for er in errors_messages]
    else:
        print(f'start \nParams: {my_namespace.__dict__}')

        from Geosteering_model import GeosteeringModel
        model = GeosteeringModel(xml_path=my_namespace.scenario_path,
                                 GR_path=my_namespace.gr_path)

        model.init_algorithm_params(my_namespace.segments_count,
                                    my_namespace.delta_deg,
                                    my_namespace.st,
                                    my_namespace.metric)
        model.start_geosteering(plot_matching=True)
        print('finish')
