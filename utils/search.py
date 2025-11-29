from utils.constants import Actions

class SearchState:
    """
    Internal search state used only by search-based agents.

    position          : current vertex (int)
    remaining_people  : tuple[int] of length n_vertices
    has_kit           : bool – whether the agent holds the kit
    """
    __slots__ = ("position", "remaining_people", "has_kit", "parent", "action_from_parent")

    def __init__(self, position, remaining_people, has_kit,
                 parent=None, action_from_parent=None):
        self.position = position
        self.remaining_people = remaining_people  # tuple
        self.has_kit = has_kit
        self.parent = parent
        self.action_from_parent = action_from_parent

    def key(self):
        # Unique key for visited set
        return (self.position, self.remaining_people, self.has_kit)

# Helper: apply the "automatic rescue" effect at a given vertex.
def apply_rescue(remaining_list, vertex):
    new_rem = remaining_list[:]        # always work on a fresh copy
    if new_rem[vertex] > 0:
        new_rem[vertex] = 0      # all people at this vertex are now rescued
    return new_rem

def successors(state, env):
    """
    Generate all legal successor states from `state`,
    by performing all possible actions and collecting all the resulting states.
    Returns a list of (next_state, (action, info), step_cost) tuples.

    - action: one of these: TRAVERSE / EQUIP / UNEQUIP / NO_OP
    - info  : the same value that env.step() expects from agents:
              * TRAVERSE -> destination vertex index (int)
              * EQUIP / UNEQUIP / NO_OP -> None
    - step_cost: time added by executing that action (used as edge cost in search)
    """
    succs = []

    pos = state.position              # current vertex
    remaining = list(state.remaining_people)  # list[int], will copy per successor
    has_kit = state.has_kit

    # 1. TRAVERSE to adjacent vertices (respect flooded edges + amphibian kit)
    for v in env.get_adjacent_vertices(pos):
        # Cannot traverse a flooded edge without holding the amphibian kit
        if env.check_flooded(pos, v) and not has_kit:
            continue

        # After moving, if there are people at v, they get rescued this step
        new_remaining = apply_rescue(remaining, v)
        new_has_kit = has_kit

        base_cost = env.weights[pos][v]
        if has_kit:
            step_cost = base_cost * env.action_duration['amphibian']
        else:
            step_cost = base_cost

        next_state = SearchState(
            position=v,
            remaining_people=tuple(new_remaining),
            has_kit=new_has_kit,
            parent=state,
            action_from_parent=(Actions.TRAVERSE, v),
        )
        succs.append((next_state, (Actions.TRAVERSE, v), step_cost))

    # 2. EQUIP (if there is a kit here and we don't already hold one)
    if not has_kit and env.check_amphibian_availability(pos):
        # After this step, if there are people here, they’ll also be rescued
        new_remaining = apply_rescue(remaining, pos)
        new_has_kit = True
        step_cost = env.action_duration['equip']

        next_state = SearchState(
            position=pos,
            remaining_people=tuple(new_remaining),
            has_kit=new_has_kit,
            parent=state,
            action_from_parent=(Actions.EQUIP, None),
        )
        succs.append((next_state, (Actions.EQUIP, None), step_cost))

    # 3. UNEQUIP (if we currently hold the kit)
    if has_kit:
        new_remaining = apply_rescue(remaining, pos)
        new_has_kit = False
        step_cost = env.action_duration['unequip']

        next_state = SearchState(
            position=pos,
            remaining_people=tuple(new_remaining),
            has_kit=new_has_kit,
            parent=state,
            action_from_parent=(Actions.UNEQUIP, None),
        )
        succs.append((next_state, (Actions.UNEQUIP, None), step_cost))

    # 4. NO_OP (stay in place for one time unit)
    # In the real env, if you stand on a P* vertex, people are rescued anyway.
    new_remaining = apply_rescue(remaining, pos)
    step_cost = 1

    next_state = SearchState(
        position=pos,
        remaining_people=tuple(new_remaining),
        has_kit=has_kit,
        parent=state,
        action_from_parent=(Actions.NO_OP, None),
    )
    succs.append((next_state, (Actions.NO_OP, None), step_cost))

    return succs
