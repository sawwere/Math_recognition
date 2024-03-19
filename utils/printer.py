import models.MathNet as mnt
import utils.letter

classes = ['(',
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
 'I-',
 'α',
 'and',
 'β',
 'δ',
 'equal',
 '∃',
 '∀',
 'γ',
 'λ',
 'μ',
 'not',
 'ω',
 'or',
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

    def print(self, letters):
        res = []
        # True sorting by Y axis
        line = 0
        
        letters.sort(key=lambda ll: (ll.y), reverse=False)  
        #print(mnt.map_pred(letters[0].value), letters[0].top, letters[0].bottom)
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
            string += classes[letter.value]
        res.append(string)
        for line in res:
            print(line)
        return res