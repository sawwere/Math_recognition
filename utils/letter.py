import numpy as np

class Letter:
    def __init__(self, x, y, w, h, img):
        self.__x = x
        self.__y = y
        self.__width = w
        self.__height = h
        self.image = img
        
        self.line = 0

        self.value = 'None'
        self.score = 0
        
    def resize():
        pass

    @property
    def x(self):
        return self.__x
    @x.setter
    def set_width(self, x):
        self.__x = x

    @property
    def y(self):
        return self.__y
    @x.setter
    def set_width(self, y):
        self.__y = y

    @property
    def width(self):
        return self.__width
    @width.setter
    def set_width(self, width : int):
        self.__width = width
    @property
    def w(self):
        return self.width
    @w.setter
    def set_w(self, width : int):
        self.width = width

    @property
    def height(self):
        return self.__height
    @height.setter
    def set_height(self, height : int):
        self.__height = height
    @property
    def h(self):
        return self.height
    @h.setter
    def set_h(self, height : int):
        self.height = height

    @property
    def left(self):
        return self.__x
    @property
    def right(self):
        return self.__x + self.__width
    @property
    def top(self):
        return self.__y
    @property
    def bottom(self):
        return self.__y + self.__height


    def add_frame(self):
        img = self.image
        size_max = max(self.width, self.height)
        letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
        if self.w > self.h:
            y_pos = size_max//2 - self.h//2
            letter_square[y_pos:y_pos + self.h, 0:self.w] = img
        elif self.w < self.h:
            x_pos = size_max//2 - self.w//2
            letter_square[0:self.h, x_pos:x_pos + self.w] = img
        else:
            letter_square = img
        self.image = letter_square

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['image']
        return state
