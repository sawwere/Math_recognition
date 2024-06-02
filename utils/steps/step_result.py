class BasicStepResult():
    def __init__(self, image) -> None:
        self.image = image
        pass

class ContourSearchStepResult(BasicStepResult):
    def __init__(self, image, letters : list, hlines : list) -> None:
        super().__init__(image)
        self.letters = letters
        self.hlines = hlines

class GroupSplittingStepResult(BasicStepResult):
    def __init__(self, image, lines : dict, hlines : list) -> None:
        super().__init__(image)
        self.lines = lines
        self.hlines = hlines

class LexerStepResult(BasicStepResult):
    def __init__(self, image, lines : dict, hlines : list) -> None:
        super().__init__(image)
        self.lines = lines
        self.hlines = hlines

class BuildTreeStepResult(BasicStepResult):
    def __init__(self, image, nodes : list) -> None:
        super().__init__(image)
        self.nodes = nodes

