
class Union:
    _i = 1000
    def __init__(self) -> None:
        Union._i += 1
        self._gid = Union._i
        self._parent = None

    def get_parent(self):
        if self._parent != None:
            self._parent = self._parent.get_parent()
            if self._parent == self: 
                self._parent = None
                return self
            return self._parent
        return self

    def grouped(self, other):
        if other.get_group() == self.get_group():
            return True
        return False

    def get_group(self):
        parent = self.get_parent()
        return parent._gid
    
    def _set_parent(self, parent):
        self.get_parent()._parent = parent

    def add_child(self, other):
        if not self.grouped(other):
            other._set_parent(self.get_parent())