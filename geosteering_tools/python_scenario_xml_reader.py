import json
import xml.etree.ElementTree
import xml.etree.ElementTree as ET
from typing import List, Optional, Union


class Point:
    def __init__(self, x: float, y: float):
        self.X = x
        self.Y = y


class Surface:
    def __init__(self, name: str):
        self.Name = name
        self.Points: List[Point] = []


class CurvePoint:
    def __init__(self, position: float, value: float):
        self.Position: float = position
        self.Value: float = value


class Curve:
    def __init__(self):
        self.Points: List[CurvePoint] = []


class TrajectoryPoint:
    def __init__(self, md: float, x: float, y: float):
        self.Md = md
        self.SectionPoint = Point(x, y)


class Property:
    def __init__(self, name: str):
        self.Name = name
        self.Real = Curve()
        self.Offset = Curve()


class Section:
    def __init__(self):
        self.Surfaces: List[Surface] = []


class Trajectory:
    def __init__(self):
        self.Points: List[TrajectoryPoint] = []


class SectionDip:
    def __init__(self, md: float, dip: float, x: float, y: float):
        self.Md: float = md
        self.Dip: float = dip
        self.Location = Point(x, y)


def is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except:
        return False


def set_value_sub_element(parent: xml.etree.ElementTree.Element, name: str, value: str, is_float_value=False):
    if (is_float_value and not is_float(value)) or value in ['', 'None', 'nan']:
        ET.SubElement(parent, name).text = ''
        return
    ET.SubElement(parent, name).text = value


def find_in_element(parent: xml.etree.ElementTree.Element, name: str, is_float_value: bool = False) \
        -> Union[str, Optional[float]]:
    value = parent.find(name).text
    if not is_float_value:
        return value
    return float(value) if is_float(value) else None


class TrajectoryMarker:
    def __init__(self, md: float, location_x: float, location_y: float, offset_position: int):
        self.Md: float = md
        self.Location = Point(location_x, location_y)
        self.OffsetPosition: int = offset_position


class Scenario:
    def __init__(self):
        self.Trajectory: Trajectory = Trajectory()
        self.Section: Section = Section()
        self.Property: List[Property] = []
        self.BeginMd: Optional[float] = None
        self.EndMd: Optional[float] = None
        self.Trajectory: Trajectory = Trajectory()
        self.Dips: List[SectionDip] = []
        self.Markers: List[TrajectoryMarker] = []

    def load(self, path: str):
        file = open(path, "r", encoding="utf-8")
        scenario_xml = ET.parse(file).getroot()

        self.BeginMd = find_in_element(scenario_xml, 'BeginMd', True)
        self.EndMd = find_in_element(scenario_xml, 'EndMd', True)

        for point_xml in list(scenario_xml.find('Trajectory').find('Points')):
            self.Trajectory.Points.append(TrajectoryPoint(find_in_element(point_xml, 'Md', True),
                                                          find_in_element(point_xml.find('SectionPoint'), 'X', True),
                                                          find_in_element(point_xml.find('SectionPoint'), 'Y', True)))

        for surface_element in list(scenario_xml.find('Section').find('Surfaces')):
            surface = Surface(find_in_element(surface_element, 'Name'))
            self.Section.Surfaces.append(surface)
            for point_element in surface_element.find('Points'):
                surface.Points.append(Point(find_in_element(point_element, 'X', True),
                                            find_in_element(point_element, 'Y', True)))

        if scenario_xml.find('Properties'):
            for property_element in list(scenario_xml.find('Properties')):
                prop = Property(find_in_element(property_element, 'Name'))
                self.Property.append(prop)

                for point_element in list(property_element.find('Real').find('Points')):
                    prop.Real.Points.append(CurvePoint(find_in_element(point_element, 'Position', True),
                                                       find_in_element(point_element, 'Value', True)))

                for point_element in list(property_element.find('Offset').find('Points')):
                    prop.Offset.Points.append(CurvePoint(find_in_element(point_element, 'Position', True),
                                                         find_in_element(point_element, 'Value', True)))

        if scenario_xml.find('Dips'):
            for section_dip_xml in list(scenario_xml.find('Dips')):
                self.Dips.append(SectionDip(find_in_element(section_dip_xml, 'Md', True),
                                            find_in_element(section_dip_xml, 'Dip', True),
                                            find_in_element(section_dip_xml.find('Location'), 'X', True),
                                            find_in_element(section_dip_xml.find('Location'), 'Y', True)))

        if scenario_xml.find('Markers'):
            for marker_xml in list(scenario_xml.find('Markers')):
                self.Markers.append(TrajectoryMarker(find_in_element(marker_xml, 'Md', True),
                                                     find_in_element(marker_xml.find('Location'), 'X', True),
                                                     find_in_element(marker_xml.find('Location'), 'Y', True),
                                                     int(find_in_element(marker_xml, 'OffsetPosition', True))))

    def save_xml(self, path: str):
        xml_file = open(path, "wb")
        et = ET.ElementTree(self.__get_data_as_xml())
        et.write(xml_file, encoding='utf-8', xml_declaration=True)

    def __get_data_as_xml(self) -> xml.etree.ElementTree.Element:
        scenario_xml = ET.Element('Scenario', attrib={'xmlns:xsd': "http://www.w3.org/2001/XMLSchema",
                                                      'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance'})

        set_value_sub_element(scenario_xml, 'BeginMd', str(self.BeginMd), True)
        set_value_sub_element(scenario_xml, 'EndMd', str(self.EndMd), True)
        set_value_sub_element(scenario_xml, 'Markers', str(self.Markers), True)

        trajectory_xml = ET.SubElement(scenario_xml, 'Trajectory')
        points_xml = ET.SubElement(trajectory_xml, 'Points')
        for trajectoryPoint in self.Trajectory.Points:
            trajectory_point_xml = ET.SubElement(points_xml, 'TrajectoryPoint')

            set_value_sub_element(trajectory_point_xml, 'Md', str(trajectoryPoint.Md), True)
            section_point_xml = ET.SubElement(trajectory_point_xml, 'SectionPoint')

            set_value_sub_element(section_point_xml, 'X', str(trajectoryPoint.SectionPoint.X), True)
            set_value_sub_element(section_point_xml, 'Y', str(trajectoryPoint.SectionPoint.Y), True)

        section_xml = ET.SubElement(scenario_xml, 'Section')
        surfaces_xml = ET.SubElement(section_xml, 'Surfaces')
        for surface in self.Section.Surfaces:
            surface_xml = ET.SubElement(surfaces_xml, 'Surface')

            set_value_sub_element(surface_xml, 'Name', str(surface.Name))

            points_surface_xml = ET.SubElement(surface_xml, 'Points')
            for surface_point in surface.Points:
                point_surface_xml = ET.SubElement(points_surface_xml, 'Point')
                set_value_sub_element(point_surface_xml, 'X', str(surface_point.X), True)
                set_value_sub_element(point_surface_xml, 'Y', str(surface_point.Y), True)

        properties_xml = ET.SubElement(scenario_xml, 'Properties')
        for prop in self.Property:
            property_xml = ET.SubElement(properties_xml, 'Property')
            set_value_sub_element(property_xml, 'Name', str(prop.Name))

            real_property_xml = ET.SubElement(property_xml, 'Real')
            offset_property_xml = ET.SubElement(property_xml, 'Offset')

            points_real_property_xml = ET.SubElement(real_property_xml, 'Points')
            points_offset_property_xml = ET.SubElement(offset_property_xml, 'Points')

            for surface_point in prop.Real.Points:
                curve_point_points_real_property_xml = ET.SubElement(points_real_property_xml, 'CurvePoint')
                set_value_sub_element(curve_point_points_real_property_xml, 'Position', str(surface_point.Position),
                                      True)
                set_value_sub_element(curve_point_points_real_property_xml, 'Value', str(surface_point.Value), True)

            for surface_point in prop.Offset.Points:
                curve_point_points_real_property_xml = ET.SubElement(points_offset_property_xml, 'CurvePoint')
                set_value_sub_element(curve_point_points_real_property_xml, 'Position', str(surface_point.Position),
                                      True)
                set_value_sub_element(curve_point_points_real_property_xml, 'Value', str(surface_point.Value), True)

        # dips_xml = ET.SubElement(scenario_xml, 'Dips')
        # for section_dip in self.Dips:
        #     section_dip_xml = ET.SubElement(dips_xml, 'SectionDip')
        #
        #     set_value_sub_element(section_dip_xml, 'Md', str(section_dip.Md), True)
        #     set_value_sub_element(section_dip_xml, 'Dip', str(section_dip.Dip), True)
        #     location_xml = ET.SubElement(section_dip_xml, 'Location')
        #
        #     set_value_sub_element(location_xml, 'X', str(section_dip.Location.X), True)
        #     set_value_sub_element(location_xml, 'Y', str(section_dip.Location.Y), True)

        markers_xml = ET.SubElement(scenario_xml, 'Markers')
        for marker in self.Markers:
            trajectory_marker_xml = ET.SubElement(markers_xml, 'TrajectoryMarker')

            set_value_sub_element(trajectory_marker_xml, 'Md', str(marker.Md), True)
            set_value_sub_element(trajectory_marker_xml, 'OffsetPosition', str(marker.OffsetPosition), True)

            location_xml = ET.SubElement(trajectory_marker_xml, 'Location')

            set_value_sub_element(location_xml, 'X', str(marker.Location.X), True)
            set_value_sub_element(location_xml, 'Y', str(marker.Location.Y), True)

        return scenario_xml


if __name__ == '__main__':
    sc1 = Scenario()
    sc1.load('C:/Users/KosachevIV/PycharmProjects/Geosteering_program/Data/AddMarkers3.xml')
    sc1.save_xml('C:/Users/KosachevIV/PycharmProjects/Geosteering_program/Data/SaveAddMarkers3.xml')

    sc2 = Scenario()
    sc2.load('C:/Users/KosachevIV/PycharmProjects/Geosteering_program/Data/SaveAddMarkers3.xml')

    print('ok')
