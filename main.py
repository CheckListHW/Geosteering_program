from geosteering_tools.arg_parser import arg_parser
from geosteering_tools.python_scenario_xml_reader import Scenario


def start(params):
    scenario = Scenario()
    scenario.load(params.scenario_path)

    from Geosteering_model import Geosteering_Model

    model = Geosteering_Model(xml_path=params.scenario_path,
                             GR_path=params.gr_path)

    print()
    print(model.th)
    print()

    model.start_complex_geosteering(min_interp_seg_size=params.min_interp_seg_size,  # используется при наличии имиджей
                                    delta_deg=params.delta_deg,
                                    st=params.st, metric=params.metric,
                                    plot_matching=False,
                                    without_dips_nof=params.segments_count)


    model.save_results_to_xml(path=params.result_path)


if __name__ == '__main__':
    my_namespace, errors_messages = arg_parser()
    if errors_messages:
        print("\n".join(errors_messages))
    else:
        print(f'Start \nParams: {my_namespace.__dict__}')
        start(my_namespace)
    print('Finish')
