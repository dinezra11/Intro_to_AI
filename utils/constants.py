class Style:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'


class Actions:
    NO_OP = 0
    TRAVERSE = 1
    EQUIP = 2
    UNEQUIP = 3

ACTIONS_STR = 'TRAVERSE 1, EQUIP 2, UNEQUIP 3, NO_OP 0'