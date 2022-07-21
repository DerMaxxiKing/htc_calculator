from uuid import UUID, uuid4
from re import sub
from .user_bcs import *


class BoundaryConditionMetaMock(type):

    cls_instances = []
    cls_instances_dict = dict()

    def get_boundary_by_name(cls, name):
        return next((x for x in cls.instances if x.name == name), None)

    def get_boundary_by_txt_id(cls, txt_id):
        return cls.cls_instances_dict[txt_id]

    def __call__(cls, *args, **kwargs):

        obj = cls.__new__(cls, *args, **kwargs)
        obj.__init__(*args, **kwargs)
        cls.cls_instances.append(obj)
        cls.cls_instances_dict[obj.txt_id] = obj
        return obj


class FaceBoundaryCondition(object, metaclass=BoundaryConditionMetaMock):

    @classmethod
    def from_block_mesh_boundary(cls, bmb):

        return cls(name=bmb.name,
                   type=bmb.type,
                   user_bc=bmb.user_bc,
                   solid_user_bc=bmb.solid_user_bc,
                   fluid_user_bc=bmb.fluid_user_bc,
                   case=bmb.case,
                   function_objects=bmb.function_objects
                   )

    def __init__(self, *args, **kwargs):

        self._txt_id = None
        self._alt_txt_id = None
        self._function_objects = None

        self.id = kwargs.get('id', uuid4().int)
        self.alt_id = kwargs.get('alt_id', uuid4().int)       # alternative id, needed in changePatchDict as existing patch type will remain the same
        self.use_alt_id = False

        self.name = kwargs.get('name')
        self.type = kwargs.get('type')
        self.faces = kwargs.get('faces', set())

        self.txt_id = kwargs.get('txt_id', None)

        self.n_faces = kwargs.get('n_faces', None)
        self.start_face = kwargs.get('start_face', None)

        self.user_bc = kwargs.get('user_bc', None)
        self.solid_user_bc = kwargs.get('solid_user_bc', None)
        self.fluid_user_bc = kwargs.get('fluid_user_bc', None)

        self.case = kwargs.get('case', None)
        self._cell_zone = kwargs.get('cell_zone', None)

        self.function_objects = kwargs.get('function_objects', [])
        self.material = kwargs.get('material', None)

    @property
    def txt_id(self):
        if self._txt_id is None:
            if isinstance(self.id, UUID):
                self._txt_id = sub('\W+', '', 'a' + str(self.id))
            else:
                self._txt_id = sub('\W+', '', 'a' + str(self.id))
        return self._txt_id

    @txt_id.setter
    def txt_id(self, value):
        self._txt_id = value

    @property
    def alt_txt_id(self):
        if self._alt_txt_id is None:
            if isinstance(self.alt_id, UUID):
                self._alt_txt_id = 'bc' + str(self.alt_id.hex)
            else:
                self._alt_txt_id = 'bc' + str(self.alt_id)
        return self._alt_txt_id

    @alt_txt_id.setter
    def alt_txt_id(self, value):
        self._alt_txt_id = value

    @property
    def faces(self):
        return self._faces

    @faces.setter
    def faces(self, value):
        self._faces = value
        [x.set_boundary(self) for x in self._faces]

    @property
    def function_objects(self):
        return self._function_objects

    @function_objects.setter
    def function_objects(self, value):
        self._function_objects = value
        for fo in self._function_objects:
            fo.patches.add(self)
            fo.cell_zone = self.cell_zone
            raise NotImplementedError

    @property
    def boundary_entry(self):

        entries = []
        for face in self.faces:
            entries.append(f'\t"{face.txt_id}"\n'
                           '\t{\n'
                           f'\t\ttype            {self.type};\n'
                           '\t}\n')

        return '\n'.join(entries)

    def field_entry(self, field_name):
        entries = []
        for face in self.faces:
            entries.append(f'\t"{self}"\n' + getattr(self.user_bc, field_name).dict_entry + '\n')

        return '\n'.join(entries)

    def __repr__(self):
        return f'Boundary {self.id} (name={self.name}, type={self.type}, faces={self.faces})'

    def __eq__(self, other):
        if not isinstance(other, FaceBoundaryCondition):
            return False
        return self.id == other.id

    def __hash__(self):
        return id(self)


class Interface(FaceBoundaryCondition):

    def __init__(self, *args, **kwargs):
        FaceBoundaryCondition.__init__(self, *args, **kwargs)
        self.face_1 = kwargs.get('face_1', None)
        self.face_2 = kwargs.get('face_2', None)


inlet = FaceBoundaryCondition(name='inlet', type='patch', user_bc=VolumeFlowInlet())
outlet = FaceBoundaryCondition(name='outlet', type='patch', user_bc=Outlet())
wall = FaceBoundaryCondition(name='wall', type='wall', user_bc=SolidWall())
fluid_wall = FaceBoundaryCondition(name='fluid_wall', type='wall', user_bc=FluidWall())
solid_wall = FaceBoundaryCondition(name='solid_wall', type='wall', user_bc=SolidWall())

top_side = FaceBoundaryCondition(name='top_side',
                                 type='wall',
                                 user_bc=SolidConvection()
                                 )

bottom_side = FaceBoundaryCondition(name='bottom_side',
                                    type='wall',
                                    user_bc=SolidConvection()
                                    )
