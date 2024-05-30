from utils.printer import PrettyPrinter    
class Group:
    def __init__(self, value) -> None:
        self.letters = value
        self.letters.sort(key=lambda ll: (ll.x), reverse=False)
        self.left = self.letters[0].left
        self.right = self.letters[-1].right
        self.top = min(map(lambda x: x.y, self.letters))
        self.bottom = max(map(lambda x: x.bottom, self.letters))
        self.line = self.letters[0].line
        self.printer = PrettyPrinter()

    def print(self):
        res = ''
        for item in self.letters:
            res += self.printer.pretty(item.value) + ' '
        print(res)

    def __getstate__(self):
            state = self.__dict__.copy()['letters']
            return state