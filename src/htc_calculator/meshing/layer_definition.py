import itertools
from copy import deepcopy
import uuid


class LayerDefinition(object):

    mapping_dict = {'n_surface_layers': 'nSurfaceLayers',                       # 0
                    'relative_sizes': 'relativeSizes',                          # 1
                    'expansion_ratio': 'expansionRatio',                        # 2
                    'first_layer_thickness': 'firstLayerThickness',             # 3
                    'final_layer_thickness': 'finalLayerThickness',             # 4
                    'thickness': 'thickness',                                   # 5
                    'min_thickness': 'minThickness',                            # 6
                    'n_grow': 'nGrow',                                          # 7
                    'n_buffer_cells_no_extrude': 'nBufferCellsNoExtrude',       # 8
                    'n_layer_iter': 'nLayerIter',                               # 9
                    'n_relaxed_iter': 'nRelaxedIter',                           # 10
                    'feature_angle': 'featureAngle' ,                           # 11
                    'slip_feature_angle': 'slipFeatureAngle',                   # 12
                    'merge_patch_faces_angle': 'mergePatchFacesAngle',          # 13
                    'max_face_thickness_ratio': 'maxFaceThicknessRatio',        # 14
                    'concave_angle': 'concaveAngle',                            # 15
                    'layer_termination_angle': 'layerTerminationAngle',         # 16
                    'n_smooth_surface_normals': 'nSmoothSurfaceNormals',        # 17
                    'n_smooth_thickness': 'nSmoothThickness',                   # 18
                    'min_medial_axis_angle': 'minMedialAxisAngle',              # 19
                    'max_thickness_to_medial_ratio': 'maxThicknessToMedialRatio',   # 20
                    'n_medial_axis_iter': 'nMedialAxisIter',                    # 21
                    'n_smooth_displacement': 'nSmoothDisplacement',             # 22
                    'detect_extrusion_island': 'detectExtrusionIsland',         # 23
                    'n_relax_iter': 'nRelaxIter',                               # 24
                    'n_smooth_normals': 'nSmoothNormals',                       # 25
                    'additional_reporting': 'additionalReporting'               # 26
                    }

    def __init__(self, *args, **kwargs):
        """
        :keyword
            * *n_surface_layers* (``str``) -- xxxxx
            * *relative_sizes* (``str``) -- xxxxx
            * *expansion_ratio* (``str``) -- xxxxx
            * *first_layer_thickness* (``str``) -- xxxxx
            * *final_layer_thickness* (``str``) -- xxxxx
            * *thickness* (``str``) -- xxxxx
            * *min_thickness* (``str``) -- xxxxx
            * *n_grow* (``str``) -- xxxxx
            * *n_buffer_cells_no_extrude* (``str``) -- xxxxx
            * *n_layer_iter* (``str``) -- xxxxx
            * *n_relaxed_iter* (``str``) -- xxxxx
            * *feature_angle* (``str``) -- xxxxx
            * *slip_feature_angle* (``str``) -- xxxxx
            * *merge_patch_faces_angle* (``str``) -- xxxxx
            * *max_face_thickness_ratio* (``str``) -- xxxxx            * *concave_angle* (``str``) -- xxxxx
            * *layer_termination_angle* (``str``) -- xxxxx
            * *n_smooth_surface_normals* (``str``) -- xxxxx
            * *n_smooth_thickness* (``str``) -- xxxxx
            * *min_medial_axis_angle* (``str``) -- xxxxx
            * *max_thickness_to_medial_ratio* (``str``) -- xxxxx
            * *n_medial_axis_iter* (``str``) -- xxxxx
            * *n_smooth_displacement* (``str``) -- xxxxx
            * *detect_extrusion_island* (``str``) -- xxxxx
            * *n_relax_iter* (``str``) -- xxxxx
            * *n_smooth_normals* (``str``) -- xxxxx
            * *additional_reporting* (``str``) -- xxxxx
        """

        self._id = None
        self._name = None

        self.id = kwargs.get('id', str(uuid.uuid4()))
        self.name = kwargs.get('_name', 'Layer Setup{}'.format(str(self.id)))

        # layer setup
        # 0
        self._n_surface_layers = kwargs.get('_n_surface_layers', kwargs.get('n_surface_layers', 0))
        # 1
        self._relative_sizes = kwargs.get('_relative_sizes', kwargs.get('relative_sizes', True))
        # 2
        self._expansion_ratio = kwargs.get('_expansion_ratio', kwargs.get('expansion_ratio', 1.2))
        # 3
        self._first_layer_thickness = kwargs.get('_first_layer_thickness', kwargs.get('first_layer_thickness', None))
        # 4
        self._final_layer_thickness = kwargs.get('_final_layer_thickness', kwargs.get('final_layer_thickness', None))
        # 5
        self._thickness = kwargs.get('_thickness', kwargs.get('thickness', None))
        # 6
        self._min_thickness = kwargs.get('_min_thickness', kwargs.get('min_thickness', 1e-5))
        # 7
        self._n_grow = kwargs.get('_n_grow', kwargs.get('n_grow', 0))
        # 8
        self._n_buffer_cells_no_extrude = kwargs.get('_n_buffer_cells_no_extrude',
                                                     kwargs.get('n_buffer_cells_no_extrude', 0))
        # 9
        self._n_layer_iter = kwargs.get('_n_layer_iter', kwargs.get('n_layer_iter', 50))
        # 10
        self._n_relaxed_iter = kwargs.get('_n_relaxed_iter', kwargs.get('n_relaxed_iter', 20))
        # 11
        self._feature_angle = kwargs.get('_feature_angle', kwargs.get('feature_angle', 90))
        # 12
        self._slip_feature_angle = kwargs.get('_slip_feature_angle', kwargs.get('slip_feature_angle', 30.0))
        # 13
        self._merge_patch_faces_angle = kwargs.get('_merge_patch_faces_angle',
                                                   kwargs.get('merge_patch_faces_angle', None))
        # 14
        self._max_face_thickness_ratio = kwargs.get('_max_face_thickness_ratio',
                                                    kwargs.get('max_face_thickness_ratio', 0.5))
        # 15
        self._concave_angle = kwargs.get('_concave_angle', kwargs.get('concave_angle', None))
        # 16
        self._layer_termination_angle = kwargs.get('_layer_termination_angle',
                                                   kwargs.get('layer_termination_angle', None))
        # 17
        self._n_smooth_surface_normals = kwargs.get('_n_smooth_surface_normals',
                                                    kwargs.get('n_smooth_surface_normals', 10))
        # 18
        self._n_smooth_thickness = kwargs.get('_n_smooth_thickness',
                                              kwargs.get('_n_smooth_thickness', 10))
        # 19
        self._min_medial_axis_angle = kwargs.get('_min_medial_axis_angle',
                                                 kwargs.get('min_medial_axis_angle', 90))
        # 20
        self._max_thickness_to_medial_ratio = kwargs.get('_max_thickness_to_medial_ratio',
                                                         kwargs.get('max_thickness_to_medial_ratio', 1.0))
        # 21
        self._n_medial_axis_iter = kwargs.get('_n_medial_axis_iter',
                                              kwargs.get('n_medial_axis_iter', None))
        # 22
        self._n_smooth_displacement = kwargs.get('_n_smooth_displacement',
                                                 kwargs.get('n_smooth_displacement', None))
        # 23
        self._detect_extrusion_island = kwargs.get('_detect_extrusion_island',
                                                   kwargs.get('_detect_extrusion_island', None))
        # 24
        self._n_relax_iter = kwargs.get('_n_relax_iter',
                                        kwargs.get('n_relax_iter', 5))
        # 25
        self._n_smooth_normals = kwargs.get('_n_smooth_normals',
                                            kwargs.get('n_smooth_normals', 3))
        # 26
        self._additional_reporting = kwargs.get('_additional_reporting', kwargs.get('additional_reporting', False))

        self._layer_setup_dict = None

        print('initialized')

    @property
    def n_surface_layers(self):
        return self._n_surface_layers

    @n_surface_layers.setter
    def n_surface_layers(self, value):
        """blah blah
        :arg
        """
        if self._n_surface_layers == value:
            return
        self._n_surface_layers = value
        self.write_layer_setup()
        print(f'n_surface_layers updated to {value}')

    @property
    def relative_sizes(self):
        return self._relative_sizes

    @relative_sizes.setter
    def relative_sizes(self, value):
        if self._relative_sizes == value:
            return
        self._relative_sizes = value
        self.write_layer_setup()
        print(f'relative_sizes updated to {value}')

    @property
    def expansion_ratio(self):
        return self._expansion_ratio

    @expansion_ratio.setter
    def expansion_ratio(self, value):
        if self._expansion_ratio == value:
            return
        self._expansion_ratio = value
        self.write_layer_setup()

    @property
    def first_layer_thickness(self):
        return self._first_layer_thickness

    @first_layer_thickness.setter
    def first_layer_thickness(self, value):
        if self._first_layer_thickness == value:
            return
        self._first_layer_thickness = value
        self.write_layer_setup()

    @property
    def final_layer_thickness(self):
        return self._final_layer_thickness

    @final_layer_thickness.setter
    def final_layer_thickness(self, value):
        if self._final_layer_thickness == value:
            return
        self._final_layer_thickness = value
        self.write_layer_setup()

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, value):
        if self._thickness == value:
            return
        self._thickness = value
        self.write_layer_setup()

    @property
    def min_thickness(self):
        return self._min_thickness

    @min_thickness.setter
    def min_thickness(self, value):
        if self._min_thickness == value:
            return
        self._min_thickness = value
        self.write_layer_setup()

    @property
    def n_grow(self):
        return self._n_grow

    @n_grow.setter
    def n_grow(self, value):
        if self._n_grow == value:
            return
        self._n_grow = value
        self.write_layer_setup()

    @property
    def n_buffer_cells_no_extrude(self):
        return self._n_buffer_cells_no_extrude

    @n_buffer_cells_no_extrude.setter
    def n_buffer_cells_no_extrude(self, value):
        if self._n_buffer_cells_no_extrude == value:
            return
        self._n_buffer_cells_no_extrude = value
        self.write_layer_setup()

    @property
    def n_layer_iter(self):
        return self._n_layer_iter

    @n_layer_iter.setter
    def n_layer_iter(self, value):
        if self._n_layer_iter == value:
            return
        self._n_layer_iter = value
        self.write_layer_setup()

    @property
    def n_relaxed_iter(self):
        return self._n_relaxed_iter

    @n_relaxed_iter.setter
    def n_relaxed_iter(self, value):
        if self._n_relaxed_iter == value:
            return
        self._n_relaxed_iter = value
        self.write_layer_setup()

    @property
    def feature_angle(self):
        return self._feature_angle

    @feature_angle.setter
    def feature_angle(self, value):
        if self._feature_angle == value:
            return
        self._feature_angle = value
        self.write_layer_setup()

    @property
    def slip_feature_angle(self):
        return self._slip_feature_angle

    @slip_feature_angle.setter
    def slip_feature_angle(self, value):
        if self._slip_feature_angle == value:
            return
        self._slip_feature_angle = value
        self.write_layer_setup()

    @property
    def merge_patch_faces_angle(self):
        return self._merge_patch_faces_angle

    @merge_patch_faces_angle.setter
    def merge_patch_faces_angle(self, value):
        if self._merge_patch_faces_angle == value:
            return
        self._merge_patch_faces_angle = value
        self.write_layer_setup()

    @property
    def max_face_thickness_ratio(self):
        return self._max_face_thickness_ratio

    @max_face_thickness_ratio.setter
    def max_face_thickness_ratio(self, value):
        if self._max_face_thickness_ratio == value:
            return
        self._max_face_thickness_ratio = value
        self.write_layer_setup()

    @property
    def concave_angle(self):
        return self._concave_angle

    @concave_angle.setter
    def concave_angle(self, value):
        if self._concave_angle == value:
            return
        self._concave_angle = value
        self.write_layer_setup()

    @property
    def layer_termination_angle(self):
        return self._layer_termination_angle

    @layer_termination_angle.setter
    def layer_termination_angle(self, value):
        if self._layer_termination_angle == value:
            return
        self._layer_termination_angle = value
        self.write_layer_setup()

    @property
    def n_smooth_surface_normals(self):
        return self._n_smooth_surface_normals

    @n_smooth_surface_normals.setter
    def n_smooth_surface_normals(self, value):
        if self._n_smooth_surface_normals == value:
            return
        self._n_smooth_surface_normals = value
        self.write_layer_setup()

    @property
    def n_smooth_thickness(self):
        return self._n_smooth_thickness

    @n_smooth_thickness.setter
    def n_smooth_thickness(self, value):
        if self._n_smooth_thickness == value:
            return
        self._n_smooth_thickness = value
        self.write_layer_setup()

    @property
    def min_medial_axis_angle(self):
        return self._min_medial_axis_angle

    @min_medial_axis_angle.setter
    def min_medial_axis_angle(self, value):
        if self._min_medial_axis_angle == value:
            return
        self._min_medial_axis_angle = value
        self.write_layer_setup()

    @property
    def max_thickness_to_medial_ratio(self):
        return self._max_thickness_to_medial_ratio

    @max_thickness_to_medial_ratio.setter
    def max_thickness_to_medial_ratio(self, value):
        if self._max_thickness_to_medial_ratio == value:
            return
        self._max_thickness_to_medial_ratio = value
        self.write_layer_setup()

    @property
    def n_medial_axis_iter(self):
        return self._n_medial_axis_iter

    @n_medial_axis_iter.setter
    def n_medial_axis_iter(self, value):
        if self._n_medial_axis_iter == value:
            return
        self._n_medial_axis_iter = value
        self.write_layer_setup()

    @property
    def n_smooth_displacement(self):
        return self._n_smooth_displacement

    @n_smooth_displacement.setter
    def n_smooth_displacement(self, value):
        if self._n_smooth_displacement == value:
            return
        self._n_smooth_displacement = value
        self.write_layer_setup()

    @property
    def detect_extrusion_island(self):
        return self._detect_extrusion_island

    @detect_extrusion_island.setter
    def detect_extrusion_island(self, value):
        if self._detect_extrusion_island == value:
            return
        self._detect_extrusion_island = value
        self.write_layer_setup()

    @property
    def n_relax_iter(self):
        return self._n_relax_iter

    @n_relax_iter.setter
    def n_relax_iter(self, value):
        if self._n_relax_iter == value:
            return
        self._n_relax_iter = value
        self.write_layer_setup()

    @property
    def n_smooth_normals(self):
        return self._n_smooth_normals

    @n_smooth_normals.setter
    def n_smooth_normals(self, value):
        if self._n_smooth_normals == value:
            return
        self._n_smooth_normals = value
        self.write_layer_setup()

    @property
    def additional_reporting(self):
        return self._additional_reporting

    @additional_reporting.setter
    def additional_reporting(self, value):
        if self._additional_reporting == value:
            return
        self._additional_reporting = value
        self.write_layer_setup()

    @property
    def layer_setup_dict(self):
        if self._layer_setup_dict is None:
            self.write_layer_setup()
        return self._layer_setup_dict

    @layer_setup_dict.setter
    def layer_setup_dict(self, value):
        if self._layer_setup_dict == value:
            return
        self._layer_setup_dict = value

    def write_layer_setup(self):
        s = '{\n'
        for attr in self.mapping_dict.keys():
            s = self.add_entry(s, attr)
        s += '}\n'

        self.layer_setup_dict = s

    def add_entry(self, s, attr):
        parameter = self.__getattribute__(attr)
        if type(parameter) is bool:
            if parameter:
                parameter = 'true'
            else:
                parameter = 'false'
        if type(parameter) is float:
            parameter = "{}".format(parameter)

        if parameter is not None:
            parameter_str = '\t{:30s}{:>20}'.format(self.mapping_dict[attr], parameter) + ';\n'
            s += parameter_str
        return s


default_layer_definition = LayerDefinition(final_layer_thickness=0.25)
default_layer_definition.n_surface_layers = 3


if __name__ == '__main__':

    new = LayerDefinition()
    new.write_layer_setup()
    print(new.layer_setup_dict)

