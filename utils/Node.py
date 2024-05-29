from utils.printer import PrettyPrinter

class Node:
    def __init__(self, value) -> None:
        self.value = value
        #self.letters.sort(key=lambda ll: (ll.x), reverse=False)
        self.x = 0#letters[0].x
        self.y = 0#min(map(lambda x: x.y, self.letters))
        self.r = 0#max(map(lambda x: x.right, self.letters))
        self.b = 0#max(map(lambda x: x.bottom, self.letters))
        self.left = None
        self.right = None
        self.parent = None
        self.printer = PrettyPrinter()

    def print(self):
        res = ''
        if self.left is not None:
            for letter in self.left.value.letters:
                res += self.printer.pretty(letter.value) + ' '
            res += ';'
            if self.right is not None:
                for letter in self.right.value.letters:
                    res += self.printer.pretty(letter.value) + ' '
            res +='\n'
            for i in range(0, 12):
                res += '--'
            res +='\n'
        for letter in self.value.letters:
            res += self.printer.pretty(letter.value) + ' '
        print(res)

    def init(self, hline, letters):
        self.letters.sort(key=lambda ll: (ll.x), reverse=False)
        self.x = letters[0].x
        self.y = min(map(lambda x: x.y, self.letters))
        self.right = max(map(lambda x: x.right, self.letters))
        self.bottom = max(map(lambda x: x.bottom, self.letters))