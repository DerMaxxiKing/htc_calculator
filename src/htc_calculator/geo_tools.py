import sys

minimum_distance = 0.2                                # Minimum distance between faces for which they are considered in contact
DetectOnlyFullSurfaceContacts = False                 # Only that faces that have contact between them on a surface whose area is non zero are detected
DetectOnlyTangencyContacts = True                     # Only that faces that have a tangential contact or can have a tangential contact if getting closer each other, with  a distance defined in "MinimumDistance", are detected. Intersected faces are excluded
DetectTangencyOnlyByLine = False                      # Among faces that have a tangential contact, only faces that have a contact between them along a line or curve, are detected (not point contact)
# If all above variable are "False", then any pair of faces closer than "MinimumDistance" are considered in contact, including intersected faces
Slope = 1000000                         # Contact stiffness
Friction = 0.3                          # Friction

tangency_error = 0.0001


def surfaces_in_contact(face_1, face_2):  # Detect if faces are in contact on a surface whose area is not zero
    _com = face_1.common(face_2)
    _result = _com.Area > 0
    _com.nullify()
    return _result


def surfaces_tangent(face_1, face_2, distance, only_by_line=False):
    # Detect if faces are in tangential contact or can be in tangential contact
    # Getting the normal on the surface through the first point closest toward another one surface
    _norm_1 = face_1.normalAt(face_1.Surface.parameter(distance[1][0][0])[0],
                              face_1.Surface.parameter(distance[1][0][0])[1])
    _norm_2 = face_2.normalAt(face_2.Surface.parameter(distance[1][0][1])[0],
                              face_2.Surface.parameter(distance[1][0][1])[1])
    #    print("Difference of tangent vectors = ", (_norm_1+_norm_2).Length)
    if (_norm_1 + _norm_2).Length < tangency_error:
        if only_by_line:
            if len(distance[1]) < 2:
                return False
            # Getting the normal on the surface through the second point closest toward another one surface
            _norm_1 = face_1.normalAt(face_1.Surface.parameter(distance[1][len(distance[1]) - 1][0])[0],
                                      face_1.Surface.parameter(distance[1][len(distance[1]) - 1][0])[1])
            _norm_2 = face_2.normalAt(face_2.Surface.parameter(distance[1][len(distance[1]) - 1][1])[0],
                                      face_2.Surface.parameter(distance[1][len(distance[1]) - 1][1])[1])
            if (_norm_1 + _norm_2).Length < tangency_error:
                return True
        else:
            return True
    return False


def search_contacts(bodiesGroup):  # Find the contacts between each surface of first body and surfaces of the rest of bodies

    sys.setrecursionlimit(5000)

    _contacts = []
    if len(bodiesGroup) <= 1:
        return _contacts
    _first_body = bodiesGroup[0]
    _rest_of_bodies = bodiesGroup[1:len(bodiesGroup)]
    _contacts = search_contacts(_rest_of_bodies)  # Recursively search contacts
    _prev_faces = len(_contacts)
    i = 0
    for _face in _first_body.Faces:  # Browse the list of shapes of the first body
        j = 0
        for _neighbour_body in _rest_of_bodies:  # Browse the list of neighbour body
            for _neighbour_face in _neighbour_body.Faces:  # browse the list of shapes for each body
                _dist = _face.distToShape(_neighbour_face)
                #                print("    ****Distance from face ", i, " of ", _first_body.Name, " to the face ", j, " of ", _neighbour_body.Name, " is ", _dist)
                if _dist[0] <= minimum_distance:
                    if DetectOnlyFullSurfaceContacts:  # Add only full surface contacts
                        if surfaces_in_contact(_face, _neighbour_face):
                            print("Adding contact due to surfaces in full contact")
                            _contacts.append([i, j])  # Appending indexes of faces in contact.Iindexes are according to number of faces checked in current instance of function
                    elif DetectOnlyTangencyContacts:  # Add only tangential contacts
                        if surfaces_in_contact(_face, _neighbour_face):
                            print("Adding contact due to surfaces tangency")
                            _contacts.append([i, j])
                    else:
                        print("Adding all contacts between surfaces closer than ", minimum_distance)
                        _contacts.append([i, j])
                j = j + 1
        i = i + 1
    for k in range(0,
                   _prev_faces):  # Correcting the indexes returned from the previous instance according to number of faces founded in the current instance
        _contacts[k][0] = _contacts[k][0] + i
        _contacts[k][1] = _contacts[k][1] + i
    for k in range(_prev_faces,
                   len(_contacts)):  # Correcting the index of faces belonging to the rest of bodies according to number of faces belonging to the first body
        _contacts[k][1] = _contacts[k][1] + i
    return _contacts


# def addContacts(facesPairs, group):  # Adding contacts according to the faces pairs given in first argument
#     _active = FemGui.getActiveAnalysis()
#     for _pair in facesPairs:
#         _obj = FreeCAD.activeDocument().addObject("Fem::ConstraintContact", "FemConstraintContact")
#         _obj.Slope = Slope
#         _obj.Friction = Friction
#         _obj.Scale = Scale
#         _obj.References = [(group, "Face" + str(_pair[0] + 1)), (group, "Face" + str(_pair[1] + 1))]
#         _active.addObject(_obj)
#
#
# def error_dialog(msg):
#     # Create a simple dialog QMessageBox
#     # The first argument indicates the icon used: one of QtGui.QMessageBox.{NoIcon, Information, Warning, Critical, Question}
#     _diag = QtGui.QMessageBox(QtGui.QMessageBox.Warning, 'Error in AutoContacts', msg)
#     _diag.setWindowModality(QtCore.Qt.ApplicationModal)
#     _diag.exec_()
#
#
# class Error(Exception):
#     """Base class for other exceptions"""
#     pass


# class ActiveAnalysis(Error):
#     """Raised when something go wrong with analyse"""
#     pass


# try:
#     if not FemGui.getActiveAnalysis():
#         raise ActiveAnalysis
#     for obj in FreeCAD.ActiveDocument.Objects:  # seach all objects in document
#         contacts = []
#         objName = obj.Name
#         objLabel = obj.Label
#         if 'Compound' in objName:  # detect if an oject is compound
#             bodiesGroup = []
#             group = App.ActiveDocument.getObject(objName).OutList  # get the list of members object
#             for member in group:
#                 if member.Shape.Faces:  # If member is having faces, then add him in group
#                     bodiesGroup.append(member)
#                 else:
#                     print("Not a Shape")
#             contacts = searchContacts(bodiesGroup)
#             if len(contacts): addContacts(contacts, obj)
# except ActiveAnalysis:
#     errorDialog("There is not an active analysis.\nPlease create a FEM analyse first!")
# except Exception:
#     print("Not object")
