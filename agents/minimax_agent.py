from agents.base_agent import BaseAgent
from utils.constants import Actions
from utils.game_state import GameState
from utils.minimax_rules import successors_game


class MinimaxAgent(BaseAgent):
    def __init__(self, id, initial_position, game_type="adversarial", max_depth=6):
        super().__init__(id, initial_position)
        self.agent_type = "Minimax"
        self.game_type = game_type
        self.max_depth = max_depth

    # =========================================================
    # Simulator hook
    # =========================================================
    def step(self, env):
        # Respect cooldown
        if self.cooldown > 0:
            self.cooldown -= 1
            return Actions.NO_OP, None

        # Build GameState from CURRENT env
        state = self._build_state_from_env(env)

        _, action = self._minimax(
            state=state,
            env=env,
            depth=0,
            alpha=float("-inf"),
            beta=float("inf"),
            visited=set()
        )

        if action is None:
            return Actions.NO_OP, None

        return action

    # =========================================================
    # Build GameState from env (CRITICAL FIX)
    # =========================================================
    def _build_state_from_env(self, env):
        remaining_people = [0] * env.n_vertices
        kit_pos = []

        for v, objs in enumerate(env.objects):
            for obj in objs:
                if obj.startswith("P"):
                    remaining_people[v] += int(obj[1:])
                elif obj == "K":
                    kit_pos.append(v)

        positions = tuple(agent.position for agent in env.agents)
        saved = tuple(agent.rescued_amount for agent in env.agents)

        return GameState(
            positions=positions,
            remaining_people=tuple(remaining_people),
            kit_pos=tuple(kit_pos),
            turn=self.id,
            time=0,
            saved=saved,
            cooldowns=tuple(agent.cooldown for agent in env.agents),
            pending=(None, None),   # env doesn’t expose pending → OK
        )

    # =========================================================
    # Minimax + AlphaBeta
    # =========================================================
    def _minimax(self, state, env, depth, alpha, beta, visited):
        if self._cutoff(state, depth, visited):
            return self._evaluate(state), None

        visited.add(state.key())
        maximizing = (state.current_player() == self.id)
        best_action = None

        if maximizing:
            value = float("-inf")
            for next_state, action, _ in successors_game(state, env):
                v, _ = self._minimax(next_state, env, depth + 1, alpha, beta, visited)
                if v > value:
                    value = v
                    best_action = action
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:
            value = float("inf")
            for next_state, action, _ in successors_game(state, env):
                v, _ = self._minimax(next_state, env, depth + 1, alpha, beta, visited)
                if v < value:
                    value = v
                    best_action = action
                beta = min(beta, value)
                if beta <= alpha:
                    break

        visited.remove(state.key())
        return value, best_action

    # =========================================================
    # Cutoff
    # =========================================================
    def _cutoff(self, state, depth, visited):
        if depth >= self.max_depth:
            return True
        if sum(state.remaining_people) == 0:
            return True
        if state.key() in visited:
            return True
        return False

    # =========================================================
    # Evaluation
    # =========================================================
    def _evaluate(self, state):
        s0, s1 = state.saved
        remaining = sum(state.remaining_people)

        if self.game_type == "adversarial":
            val = (s0 - s1) if self.id == 0 else (s1 - s0)

        elif self.game_type == "semi":
            own = s0 if self.id == 0 else s1
            other = s1 if self.id == 0 else s0
            val = own + 0.001 * other

        elif self.game_type == "cooperative":
            val = s0 + s1

        else:
            raise ValueError("Unknown game type")

        val -= 0.01 * remaining
        return val
