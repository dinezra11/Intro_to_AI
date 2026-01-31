from utils.constants import Actions

class GameState:
    """
    A pure snapshot of the game for minimax (Assignment 2).

    It represents EVERYTHING that matters for future decisions:
    - both agents locations
    - remaining people per vertex
    - where the kits are (on ground or carried)
    - whose turn it is
    - time passed (for deadline D)
    - each agent’s saved count (IS1, IS2)

    Optional: parent/action_from_parent help debugging or plan reconstruction,
    but minimax usually only needs the best root action.
    """

    __slots__ = (
        "positions",          # (pos0, pos1)
        "remaining_people",   # tuple[int] length N: counts per vertex
        "kit_pos",            # tuple[int] length K: each kit location (>=0) or carried (-1/-2). 
                              # kit_pos[k] >= 0 means kit k lies on vertex kit_pos[k]
                              # kit_pos[k] == -1 means carried by agent 0. kit_pos[k] == -2 means carried by agent 1. 
        "turn",               # 0 or 1 (whose move)
        "time",               # int time elapsed
        "saved",              # (saved0, saved1) = individual scores IS
        "parent",
        "action_from_parent",
        "cooldowns",
        "pending"
    )

    def __init__(self, positions, remaining_people, kit_pos, turn, time, saved, cooldowns=(0, 0), pending=(None, None),
                 parent=None, action_from_parent=None):
        self.positions = positions                  # tuple of 2 ints
        self.remaining_people = remaining_people    # tuple of ints
        self.kit_pos = kit_pos                      # tuple of ints
        self.turn = turn                            # 0 or 1
        self.time = time                            # int
        self.saved = saved                          # tuple of 2 ints
        self.cooldowns = cooldowns
        self.pending = pending
        self.parent = parent
        self.action_from_parent = action_from_parent

    # ---------- Helpers (used by minimax engine) ----------

    def key(self):
        """
        Unique identity of a 'world state' for repetition detection.
        The assignment says revisiting a world state ends the game.
        So we include everything that defines the world + whose turn + time.
        """
        return (self.positions, self.remaining_people, self.kit_pos, self.turn, self.time, self.cooldowns, self.pending)

    def current_player(self):
        return self.turn

    def other_player(self):
        return 1 - self.turn

    def is_carrying_kit(self, player_id):
        """
        Player carries a kit if any kit_pos equals -(player_id+1):
        -1 means carried by player 0
        -2 means carried by player 1
        """
        token = -(player_id + 1)
        return token in self.kit_pos
