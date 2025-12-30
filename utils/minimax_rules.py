from utils.constants import Actions
from utils.game_state import GameState


def build_initial_gamestate(env, agent0, agent1):
    """
    Build the minimax GameState from the actual environment.

    agent0, agent1 are the two agent objects that already exist in your env
    (so we can read their starting positions and kit status).
    """

    n = env.n_vertices
    remaining_people = [0] * n
    kit_pos = []

    for v, objs in enumerate(env.objects):
        for obj in objs:
            if obj.startswith("P"):
                # example: "P3" -> 3 people
                remaining_people[v] += int(obj[1:])
            elif obj == "K":
                # each "K" is one kit object on the ground at this vertex
                kit_pos.append(v)

    # positions of both agents
    positions = (agent0.position, agent1.position)
    # saved scores so far (IS0, IS1)
    saved = (0, 0)
    # whose turn: in minimax we usually start with agent 0 as MAX
    turn = 0
    # time elapsed
    time = 0

    return GameState(
        positions=positions,
        remaining_people=tuple(remaining_people),
        kit_pos=tuple(kit_pos),
        turn=turn,
        time=time,
        saved=saved,
        parent=None,
        action_from_parent=None,
    )



def _player_token(player_id: int) -> int:
    # kit_pos entry == -1 means carried by player 0; -2 by player 1
    return -(player_id + 1)

def _reserve_token(player_id: int) -> int:
    # -3 for player 0, -4 for player 1
    return -(player_id + 3)

def _is_reserved_by(kit_pos_value: int, player_id: int) -> bool:
    return kit_pos_value == _reserve_token(player_id)



def _find_ground_kit_at_vertex(kit_pos, vertex: int):
    """Return index of some kit lying on `vertex`, else None."""
    for i, kp in enumerate(kit_pos):
        if kp == vertex:
            return i
    return None


def _find_carried_kit(kit_pos, player_id: int):
    """Return index of the kit carried by player_id, else None."""
    token = _player_token(player_id)
    for i, kp in enumerate(kit_pos):
        if kp == token:
            return i
    return None


def _apply_rescue(remaining_people, saved, player_id: int, vertex: int):
    """Rescue all people at `vertex` and credit them to player_id."""
    if remaining_people[vertex] <= 0:
        return remaining_people, saved

    rescued = remaining_people[vertex]
    new_remaining = list(remaining_people)
    new_remaining[vertex] = 0

    s0, s1 = saved
    if player_id == 0:
        s0 += rescued
    else:
        s1 += rescued

    return tuple(new_remaining), (s0, s1)


def _complete_pending_if_needed(state: GameState, env):
    """
    If the current player is busy, we advance one time unit (one turn),
    decrement cooldown, and if it reaches 0 we apply the pending effect.
    Returns the single forced successor state (no action choice), or None if not busy.
    """
    p = state.turn
    cd0, cd1 = state.cooldowns
    pend0, pend1 = state.pending

    cds = [cd0, cd1]
    pends = [pend0, pend1]

    if cds[p] <= 0:
        return None  # not busy

    # Spend this turn continuing the action (no choice)
    cds[p] -= 1

    positions = list(state.positions)
    kit_pos = list(state.kit_pos)
    remaining_people = state.remaining_people
    saved = state.saved

    # If action completes now, apply its effect
    if cds[p] == 0 and pends[p] is not None:
        kind, arg = pends[p]

        if kind == "MOVE":
            dest = arg
            positions[p] = dest
            # After arriving, rescue happens automatically before next move
            remaining_people, saved = _apply_rescue(remaining_people, saved, p, dest)

        elif kind == "EQUIP":
            kit_i = arg
            # Equip completes now. The kit MUST currently be reserved by this player.
            # Convert reservation (-3/-4) to carried (-1/-2).
            kit_pos[kit_i] = _player_token(p)

        elif kind == "UNEQUIP":
            kit_i = arg
            # unequip drops kit on current vertex
            kit_pos[kit_i] = positions[p]

        pends[p] = None

    next_state = GameState(
        positions=tuple(positions),
        remaining_people=remaining_people,
        kit_pos=tuple(kit_pos),
        turn=1 - p,                      # switch turn
        time=state.time + 1,             # 1 time unit passes each turn
        saved=saved,
        cooldowns=(cds[0], cds[1]),
        pending=(pends[0], pends[1]),
        parent=state,
        action_from_parent=(Actions.NO_OP, None),  # “forced continue” is like no-op
    )

    # step_cost is always 1 per turn in this assignment
    return [(next_state, (Actions.NO_OP, None), 1)]


def successors_game(state: GameState, env):
    """
    Generate all legal successors for the current player:

    - Edge weights are 1 (so traverse without kit takes 1 turn if succeeds).
    - If traverse on a flooded edge without an equipped kit -> fails and behaves like no-op (takes 1).
    - If traverse with equipped kit -> the agent is “on the edge” for P turns (amphibian factor),
      i.e., movement completes after P turns and cannot be aborted.
    - EQUIP takes Q turns, UNEQUIP takes U turns, cannot be aborted.
    - Each turn consumes 1 time unit and then turn switches.
    - When arriving at a vertex, the agent rescues all people there automatically.
    """
    # 1) If busy, there is exactly ONE forced successor
    forced = _complete_pending_if_needed(state, env)
    if forced is not None:
        return forced

    p = state.turn
    pos = state.positions[p]

    Q = env.action_duration["equip"]
    U = env.action_duration["unequip"]
    P = env.action_duration["amphibian"]  # with kit, traverse lasts P turns (even though edge weight=1)

    carried_kit = _find_carried_kit(state.kit_pos, p)

    succs = []

    # ---------- NO_OP ----------
    succs.append((
        GameState(
            positions=state.positions,
            remaining_people=state.remaining_people,
            kit_pos=state.kit_pos,
            turn=1-p,
            time=state.time + 1,
            saved=state.saved,
            cooldowns=state.cooldowns,
            pending=state.pending,
            parent=state,
            action_from_parent=(Actions.NO_OP, None),
        ),
        (Actions.NO_OP, None),
        1
    ))

    # ---------- EQUIP ----------
    # Allowed if there is a kit on the ground at current vertex AND player carries none
    if carried_kit is None:
        kit_i = _find_ground_kit_at_vertex(state.kit_pos, pos)
        if kit_i is not None:
            # Start equip now: current turn spent, remaining Q-1 future turns
            cds = list(state.cooldowns)
            pends = list(state.pending)

            kit_pos = list(state.kit_pos)

            # Reserve immediately
            kit_pos[kit_i] = _reserve_token(p)  # mark that the kit is being equipped right now, so no other agent can start equpping

            # Spend this turn now; remaining Q-1 future turns busy
            cds[p] = max(Q - 1, 0)
            pends[p] = ("EQUIP", kit_i)

            succs.append((
                GameState(
                    positions=state.positions,
                    remaining_people=state.remaining_people,
                    kit_pos=tuple(kit_pos),
                    turn=1-p,
                    time=state.time + 1,
                    saved=state.saved,
                    cooldowns=(cds[0], cds[1]),
                    pending=(pends[0], pends[1]),
                    parent=state,
                    action_from_parent=(Actions.EQUIP, None),
                ),
                (Actions.EQUIP, None),
                1
            ))

    # ---------- UNEQUIP ----------
    # Allowed if player carries a kit
    if carried_kit is not None:
        cds = list(state.cooldowns)
        pends = list(state.pending)
        cds[p] = max(U - 1, 0)
        pends[p] = ("UNEQUIP", carried_kit)

        succs.append((
            GameState(
                positions=state.positions,
                remaining_people=state.remaining_people,
                kit_pos=state.kit_pos,
                turn=1-p,
                time=state.time + 1,
                saved=state.saved,
                cooldowns=(cds[0], cds[1]),
                pending=(pends[0], pends[1]),
                parent=state,
                action_from_parent=(Actions.UNEQUIP, None),
            ),
            (Actions.UNEQUIP, None),
            1
        ))

    # ---------- TRAVERSE ----------
    # Generate neighbors
    neighbors = env.get_adjacent_vertices(pos)
    
    for v in neighbors:
        # Determine if edge is flooded
        flooded = False
        flooded = env.check_flooded(pos, v)

        if flooded and carried_kit is None:
            # traverse fails -> behaves like NO_OP (takes 1 time unit)
            # (we could skip because NO_OP already exists, but keeping it explicit is clearer)
            succs.append((
                GameState(
                    positions=state.positions,
                    remaining_people=state.remaining_people,
                    kit_pos=state.kit_pos,
                    turn=1-p,
                    time=state.time + 1,
                    saved=state.saved,
                    cooldowns=state.cooldowns,
                    pending=state.pending,
                    parent=state,
                    action_from_parent=(Actions.TRAVERSE, v),  # attempted traverse
                ),
                (Actions.TRAVERSE, v),
                1
            ))
            continue

        # traverse succeeds
        if carried_kit is None:
            # completes immediately in 1 turn (edge weight = 1)
            positions = list(state.positions)
            positions[p] = v

            remaining_people, saved = _apply_rescue(state.remaining_people, state.saved, p, v)

            succs.append((
                GameState(
                    positions=tuple(positions),
                    remaining_people=remaining_people,
                    kit_pos=state.kit_pos,
                    turn=1-p,
                    time=state.time + 1,
                    saved=saved,
                    cooldowns=state.cooldowns,
                    pending=state.pending,
                    parent=state,
                    action_from_parent=(Actions.TRAVERSE, v),
                ),
                (Actions.TRAVERSE, v),
                1
            ))
        else:
            # with kit: movement takes P turns total.
            # Spend current turn now; remaining P-1 future turns busy, then arrive at v.
            cds = list(state.cooldowns)
            pends = list(state.pending)
            cds[p] = max(P - 1, 0)
            pends[p] = ("MOVE", v)

            succs.append((
                GameState(
                    positions=state.positions,              # still at pos until move completes
                    remaining_people=state.remaining_people,
                    kit_pos=state.kit_pos,
                    turn=1-p,
                    time=state.time + 1,
                    saved=state.saved,
                    cooldowns=(cds[0], cds[1]),
                    pending=(pends[0], pends[1]),
                    parent=state,
                    action_from_parent=(Actions.TRAVERSE, v),
                ),
                (Actions.TRAVERSE, v),
                1
            ))

    return succs