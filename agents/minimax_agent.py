from __future__ import annotations

from typing import Optional, Tuple, Dict, Any

from agents.base_agent import BaseAgent
from utils.constants import Actions
from utils.game_state import GameState
from utils.minimax_rules import successors_game

ActionTuple = Tuple[int, Optional[int]]  # (Actions.*, info)


class MinimaxAgent(BaseAgent):
    """Depth-limited minimax + alpha-beta agent for Assignment 2."""

    def __init__(self, id: int, initial_position: int, game_type: str = "adversarial", max_depth: int = 6):
        super().__init__(id, initial_position)
        self.agent_type = "minimax"
        self.game_type = game_type  # "adversarial" | "semi" | "cooperative"
        self.max_depth = max_depth

    # =========================================================
    # Simulator hook
    # =========================================================
    def step(self, env) -> ActionTuple:
        """Called only on this agent's turn when env.turn_based=True."""

        # Environment already decrements cooldowns each tick.
        if self.cooldown > 0:
            return Actions.NO_OP, None

        root = self._build_state_from_env(env)

        # Per-decision transposition table (fast + stable)
        tt: Dict[Tuple[Any, int], float] = {}
        path = set()

        value, action = self._minimax(
            root, env, depth=0,
            alpha=float("-inf"), beta=float("inf"),
            path=path, tt=tt
        )

        if action is None:
            return Actions.NO_OP, None
        return action

    # =========================================================
    # Root state construction (from the REAL env)
    # =========================================================
    def _build_state_from_env(self, env) -> GameState:
        remaining_people = [0] * env.n_vertices
        kit_pos = []

        for v, objs in enumerate(env.objects):
            for obj in objs:
                if isinstance(obj, str) and obj.startswith("P"):
                    remaining_people[v] += int(obj[1:])
                elif obj == "K":
                    kit_pos.append(v)

        # Kits carried by agents are stored on the agent object (not in env.objects)
        for a in env.agents:
            if getattr(a, "is_holding_amphibian", False):
                kit_pos.append(-1 if a.id == 0 else -2)

        positions = tuple(a.position for a in env.agents)
        saved = tuple(getattr(a, "rescued_amount", 0) for a in env.agents)
        time = getattr(env, "steps", 0)

        return GameState(
            positions=positions,
            remaining_people=tuple(remaining_people),
            kit_pos=tuple(kit_pos),
            turn=getattr(env, "turn", self.id),   # IMPORTANT: use env.turn
            time=time,
            saved=saved,
            cooldowns=(0, 0),                     # root starts clean (env doesn't expose pending)
            pending=(None, None),
            parent=None,
            action_from_parent=None,
        )

    # =========================================================
    # Minimax + AlphaBeta
    # =========================================================
    def _minimax(
        self,
        state: GameState,
        env,
        depth: int,
        alpha: float,
        beta: float,
        path: set,
        tt: Dict[Tuple[Any, int], float],
    ) -> Tuple[float, Optional[ActionTuple]]:

        if self._cutoff(state, depth, path):
            return self._evaluate(state, env), None

        key = state.key()
        maximizing = (state.current_player() == self.id)

        tt_key = (key, depth)
        if tt_key in tt:
            return tt[tt_key], None

        path.add(key)

        best_action = None
        if maximizing:
            value = float("-inf")
            for next_state, action, _ in successors_game(state, env):
                v, _ = self._minimax(next_state, env, depth + 1, alpha, beta, path, tt)
                if v > value:
                    value = v
                    best_action = action
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:
            value = float("inf")
            for next_state, action, _ in successors_game(state, env):
                v, _ = self._minimax(next_state, env, depth + 1, alpha, beta, path, tt)
                if v < value:
                    value = v
                    best_action = action
                beta = min(beta, value)
                if beta <= alpha:
                    break

        path.remove(key)
        tt[tt_key] = value
        return value, best_action

    # =========================================================
    # Cutoff
    # =========================================================
    def _cutoff(self, state: GameState, depth: int, path: set) -> bool:
        if depth >= self.max_depth:
            return True
        if sum(state.remaining_people) == 0:
            return True
        if state.key() in path:
            return True
        return False

    # =========================================================
    # Evaluation
    # =========================================================
    def _evaluate(self, state: GameState, env) -> float:
        s0, s1 = state.saved

        # Approximate Environment scoring:
        # each global step costs every agent -1 score, and each rescued person gives +1000.
        score0 = 1000 * s0 - state.time
        score1 = 1000 * s1 - state.time

        # Base utility by game type
        if self.game_type == "adversarial":
            base = (score0 - score1) if self.id == 0 else (score1 - score0)
        elif self.game_type == "semi":
            own = score0 if self.id == 0 else score1
            other = score1 if self.id == 0 else score0
            base = own + 0.001 * other
        elif self.game_type == "cooperative":
            base = score0 + score1
        else:
            raise ValueError("Unknown game_type")

        # IMPORTANT tie-breaker:
        # In adversarial score-difference, time cancels out; this tiny bias ensures
        # "finish earlier" beats "NO_OP now, do it later".
        base += -0.01 * state.time

        if sum(state.remaining_people) == 0:
            return float(base)

        dist_mat = getattr(env, "optimistic_dist", None)

        def nearest_dist(player_id: int) -> float:
            start = state.positions[player_id]
            best = float("inf")
            for v, cnt in enumerate(state.remaining_people):
                if cnt <= 0:
                    continue
                d = dist_mat[start][v] if dist_mat is not None else abs(start - v)
                if d < best:
                    best = d
            return best if best != float("inf") else 1e9

        d0 = nearest_dist(0)
        d1 = nearest_dist(1)

        # small progress heuristic (only to break ties when rescue is equal)
        if self.game_type == "adversarial":
            prog = (-d0 + d1) if self.id == 0 else (-d1 + d0)
        elif self.game_type == "semi":
            prog = -d0 if self.id == 0 else -d1
        else:
            prog = -(d0 + d1)

        remaining_total = sum(state.remaining_people)
        finish_pressure = -0.1 * remaining_total

        return float(base + 0.2 * prog + finish_pressure)
