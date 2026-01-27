
class Roi(object):
    def __init__(self, y_start, y_end, x_start, x_end, name=None):
        self.y_start = y_start
        self.y_end = y_end
        self.x_start = x_start
        self.x_end = x_end
        self.name = name

    def as_tuple(self):
        return (self.y_start, self.y_end, self.x_start, self.x_end)