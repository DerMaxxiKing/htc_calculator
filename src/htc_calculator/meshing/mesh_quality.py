import itertools
from copy import deepcopy
import uuid


class MeshQuality(object):

    mapping_dict = {'_max_non_ortho': 'maxNonOrtho',
                    '_max_boundary_skewness': 'maxBoundarySkewness',
                    '_max_internal_skewness': 'maxInternalSkewness',
                    '_max_concave': 'maxConcave',
                    '_min_flatness': 'minFlatness',
                    '_min_vol': 'minVol',
                    '_min_tet_quality': 'minTetQuality',
                    '_min_area': 'minArea',
                    '_min_twist': 'minTwist',
                    '_min_determinant': 'minDeterminant',
                    '_min_face_weight': 'minFaceWeight',
                    '_min_vol_ratio': 'minVolRatio',
                    '_min_triangle_twist': 'minTriangleTwist',
                    '_n_smooth_scale': 'nSmoothScale',
                    '_error_reduction': 'errorReduction',
                    '_relaxed': 'relaxed',
                    }

    def __init__(self, *args, **kwargs):
        """
        :keyword
            * *max_non_ortho* (``str``) -- xxxxx
            * *max_boundary_skewness* (``str``) -- xxxxx
            * *max_internal_skewness* (``str``) -- xxxxx
            * *max_concave* (``str``) -- xxxxx
            * *min_flatness* (``str``) -- xxxxx
            * *min_vol* (``str``) -- xxxxx
            * *min_tet_quality* (``str``) -- xxxxx
            * *min_area* (``str``) -- xxxxx
            * *min_twist* (``str``) -- xxxxx
            * *min_determinant* (``str``) -- xxxxx
            * *min_face_weight* (``str``) -- xxxxx
            * *min_vol_ratio* (``str``) -- xxxxx
            * *min_triangle_twist* (``str``) -- xxxxx
            * *n_smooth_scale* (``str``) -- xxxxx
            * *error_reduction* (``str``) -- xxxxx
            * *relaxed* (``str``) -- xxxxx
        """

        self._id = kwargs.get('_id', kwargs.get('id', uuid.uuid4()))
        self._name = kwargs.get('_name', kwargs.get('name', 'SnappyHexMesh {}'.format(self._id)))

        self._name = kwargs.get('_name', 'Mesh Quality {}'.format(self._id))
        self._mesh_quality_dict = None

        # -----------------------------------------
        # attribute definition
        # -----------------------------------------

        self._max_non_ortho = kwargs.get('_max_non_ortho', kwargs.get('max_non_ortho', 65.0))

        self._max_boundary_skewness = kwargs.get('_max_boundary_skewness', kwargs.get('max_boundary_skewness', 20.0))

        self._max_internal_skewness = kwargs.get('_max_internal_skewness', kwargs.get('max_internal_skewness', 4.0))

        self._max_concave = kwargs.get('_max_concave', kwargs.get('max_concave', 80.0))

        self._min_flatness = kwargs.get('_min_flatness', kwargs.get('min_flatness', None))

        self._min_vol = kwargs.get('_min_vol', kwargs.get('min_vol', 1e-13))

        self._min_tet_quality = kwargs.get('_min_tet_quality', kwargs.get('min_tet_quality', 1e-15))

        self._min_area = kwargs.get('_min_area', kwargs.get('min_area', -1.0))

        self._min_twist = kwargs.get('_min_twist', kwargs.get('min_twist', 0.02))

        self._min_determinant = kwargs.get('_min_determinant', kwargs.get('min_determinant', 0.001))

        self._min_face_weight = kwargs.get('_min_face_weight', kwargs.get('min_face_weight', 0.05))

        self._min_vol_ratio = kwargs.get('_min_vol_ratio', kwargs.get('min_vol_ratio', 0.01))

        self._min_triangle_twist = kwargs.get('_min_triangle_twist', kwargs.get('min_triangle_twist', -1.0))

        self._n_smooth_scale = kwargs.get('_n_smooth_scale', kwargs.get('n_smooth_scale', 4))

        self._error_reduction = kwargs.get('_error_reduction', kwargs.get('error_reduction', 0.75))

        relaxed = (
            '{\n'
                '\tmaxNonOrtho 75;\n'
            '}\n'
        )
        self._relaxed = kwargs.get('_relaxed', kwargs.get('relaxed', relaxed))

    # -----------------------------------------
    # property setup for attributes
    # -----------------------------------------

    @property
    def max_non_ortho(self):
        return self._max_non_ortho

    @max_non_ortho.setter
    def max_non_ortho(self, value):
        if self._max_non_ortho == value:
            return
        self._max_non_ortho = value
        self.create_mesh_quality_dict()

    @property
    def max_boundary_skewness(self):
        return self._max_boundary_skewness

    @max_boundary_skewness.setter
    def max_boundary_skewness(self, value):
        if self._max_boundary_skewness == value:
            return
        self._max_boundary_skewness = value
        self.create_mesh_quality_dict()

    @property
    def max_internal_skewness(self):
        return self._max_internal_skewness

    @max_internal_skewness.setter
    def max_internal_skewness(self, value):
        if self._max_internal_skewness == value:
            return
        self._max_internal_skewness = value
        self.create_mesh_quality_dict()

    @property
    def max_concave(self):
        return self._max_concave

    @max_concave.setter
    def max_concave(self, value):
        if self._max_concave == value:
            return
        self._max_concave = value
        self.create_mesh_quality_dict()

    @property
    def min_flatness(self):
        return self._min_flatness

    @min_flatness.setter
    def min_flatness(self, value):
        if self._min_flatness == value:
            return
        self._min_flatness = value
        self.create_mesh_quality_dict()

    @property
    def min_vol(self):
        return self._min_vol

    @min_vol.setter
    def min_vol(self, value):
        if self._min_vol == value:
            return
        self._min_vol = value
        self.create_mesh_quality_dict()

    @property
    def min_tet_quality(self):
        return self._min_tet_quality

    @min_tet_quality.setter
    def min_tet_quality(self, value):
        if self._min_tet_quality == value:
            return
        self._min_tet_quality = value
        self.create_mesh_quality_dict()

    @property
    def min_area(self):
        return self._min_area

    @min_area.setter
    def min_area(self, value):
        if self._min_area == value:
            return
        self._min_area = value
        self.create_mesh_quality_dict()
    @property
    def min_twist(self):
        return self._min_twist

    @min_twist.setter
    def min_twist(self, value):
        if self._min_twist == value:
            return
        self._min_twist = value
        self.create_mesh_quality_dict()

    @property
    def min_determinant(self):
        return self._min_determinant

    @min_determinant.setter
    def min_determinant(self, value):
        if self._min_determinant == value:
            return
        self._min_determinant = value
        self.create_mesh_quality_dict()

    @property
    def min_face_weight(self):
        return self._min_face_weight

    @min_face_weight.setter
    def min_face_weight(self, value):
        if self._min_face_weight == value:
            return
        self._min_face_weight = value
        self.create_mesh_quality_dict()

    @property
    def min_vol_ratio(self):
        return self._min_vol_ratio

    @min_vol_ratio.setter
    def min_vol_ratio(self, value):
        if self._min_vol_ratio == value:
            return
        self._min_vol_ratio = value
        self.create_mesh_quality_dict()

    @property
    def min_triangle_twist(self):
        return self._min_triangle_twist

    @min_triangle_twist.setter
    def min_triangle_twist(self, value):
        if self._min_triangle_twist == value:
            return
        self._min_triangle_twist = value
        self.create_mesh_quality_dict()

    @property
    def n_smooth_scale(self):
        return self._n_smooth_scale

    @n_smooth_scale.setter
    def n_smooth_scale(self, value):
        if self._n_smooth_scale == value:
            return
        self._n_smooth_scale = value
        self.create_mesh_quality_dict()

    @property
    def error_reduction(self):
        return self._error_reduction

    @error_reduction.setter
    def error_reduction(self, value):
        if self._error_reduction == value:
            return
        self._error_reduction = value
        self.create_mesh_quality_dict()

    @property
    def relaxed(self):
        return self._relaxed

    @relaxed.setter
    def relaxed(self, value):
        if self._relaxed == value:
            return
        self._relaxed = value
        self.create_mesh_quality_dict()

    @property
    def mesh_quality_dict(self):
        if self._mesh_quality_dict is None:
            self.create_mesh_quality_dict()
        return self._mesh_quality_dict

    @mesh_quality_dict.setter
    def mesh_quality_dict(self, value):
        if self._mesh_quality_dict == value:
            return
        self._mesh_quality_dict = value

    def create_mesh_quality_dict(self):

        s = '{\n'
        for attr in self.mapping_dict.keys():
            s = self.add_entry(s, attr)
        s += '}\n'

        self.mesh_quality_dict = s

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

    def dict_entry(self):
        return self.mesh_quality_dict


default_mesh_quality = MeshQuality()

