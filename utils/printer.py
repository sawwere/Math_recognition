import models.MathNet as mnt
import utils.letter

classes = ['(', ')','+','-',
 '0','1','2', '3','4','5','6','7','8','9',
 '=',
 'A','C','F',
 'I-',
 'alpha',
 'and',
 'beta',
 'delta',
 'equal',
 'exists',
 'forall',
 'gamma',
 '>',
 'lambda',
 'mu',
 'not',
 'omega',
 'or',
 'phi',
 'pi',
 'psi',
 'rightarrow',
 'sigma',
 'tau',
 'theta',
 'upsilon',
 'x',
 'y',
 'z',
 'Г']

pretty_classes = ['(',
 ')',
 '+',
 '-',
 '0',
 '1',
 '2',
 '3',
 '4',
 '5',
 '6',
 '7',
 '8',
 '9',
 '=',
 'A',
 'C',
 'F',
 '├',
 'α',
 '⋀',
 'β',
 'δ',
 '≡',
 '∃',
 '∀',
 'γ',
 '>',
 'λ',
 'μ',
 '¬',
 'ω',
 '⋁',
 'φ',
 'pi',
 'ψ',
 '→',
 'σ',
 'τ',
 'θ',
 'υ',
 'x',
 'y',
 'z',
 'Г']

class PrettyPrinter:
    def __init__(self):
        pass

    def char(self, idx):
        return classes[idx]
    
    def pretty(self, idx):
        return pretty_classes[idx]

    def print(self, letters, need_to_sort=True):
        res = []
        # True sorting by Y axis
        line = 0
        
        
        #print(mnt.map_pred(letters[0].value), letters[0].top, letters[0].bottom)
        if (need_to_sort):
            letters.sort(key=lambda ll: (ll.y), reverse=False)  
            for i in range (1, len(letters)):
                if letters[i].top > letters[i-1].bottom:
                    line += 1
                letters[i].line = line
                #print(mnt.map_pred(letters[i].value), letters[i].top, letters[i-1].bottom, line)
            letters.sort(key=lambda ll: (ll.line, ll.x), reverse=False)  

        prev_line=-1
        string = ""
        for letter in letters:
            string += " " 
            #print(mnt.map_pred(letter.value), letter.top, letter.bottom, letter.line)
            if (letter.line >  prev_line):
                res.append(string)
                string = ""
                prev_line = letter.line
            string += pretty_classes[letter.value]
        res.append(string)
        for line in res:
            print(line)
        return res