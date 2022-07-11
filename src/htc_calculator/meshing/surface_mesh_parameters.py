import logging
import uuid


class SurfaceMeshParameters(object):

    def __init__(self, *args, **kwargs):
        """
        :keyword
            * *add_layers* (``str``) -- xxxxx
            * *layer_definition* (``str``) -- xxxxx
            * *min_refinement_level* (``str``) -- xxxxx
            * *max_refinement_level* (``str``) -- xxxxx
        """

        self._id = None
        self._name = None

        self.id = kwargs.get('id', str(uuid.uuid4()))
        self.name = kwargs.get('name', 'Base{}'.format(str(self.id)))

        self._is_default = False

        self._add_layers = kwargs.get('_add_layers', kwargs.get('add_layers', False))
        self._layer_definition = kwargs.get('_layer_definition', kwargs.get('layer_definition', None))

        self._min_refinement_level = kwargs.get('_min_refinement_level', kwargs.get('min_refinement_level', 0))
        self._max_refinement_level = kwargs.get('_max_refinement_level', kwargs.get('max_refinement_level', 2))

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def add_layers(self):
        return self._add_layers

    @add_layers.setter
    def add_layers(self, value):
        if self._add_layers == value:
            return
        self._add_layers = value

    @property
    def layer_definition(self):
        return self._layer_definition

    @layer_definition.setter
    def layer_definition(self, value):
        if self._layer_definition == value:
            return
        self._layer_definition = value

    @property
    def min_refinement_level(self):
        return self._min_refinement_level

    @min_refinement_level.setter
    def min_refinement_level(self, value):
        if self._min_refinement_level == value:
            return
        if value > self.max_refinement_level:
            logging.error(f'{self.name}: min refinement level must not be higher than max refinement level')
            return
        self._min_refinement_level = value

    @property
    def max_refinement_level(self):
        return self._max_refinement_level

    @max_refinement_level.setter
    def max_refinement_level(self, value):
        if self._max_refinement_level == value:
            return
        if value < self.min_refinement_level:
            logging.error(f'{self.name}: max refinement level must not be lower than min refinement level')
            return
        self._max_refinement_level = value

    @property
    def is_default(self):
        return self._is_default

    @is_default.setter
    def is_default(self, value):
        if self._is_default == value:
            return
        self._is_default = value


default_surface_mesh_parameter = SurfaceMeshParameters(name='default_surface_mesh_setup',
                                                       min_refinement_level=0,
                                                       max_refinement_level=4)
