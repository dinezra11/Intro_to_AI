from agents.base_agent import BaseAgent
from utils.constants import Actions
from utils.game_state import GameState
from utils.minimax_rules import build_initial_gamestate, successors_game


class MinimaxAgent(BaseAgent):
    def __init__(self, id, initial_position, game_type="adversarial", max_depth=6):
        super().__init__(id, initial_position)
        self.agent_type = "Minimax"
        self.game_type = game_type        # "adversarial" | "semi" | "cooperative"
        self.max_depth = max_depth

    # =========================================================
    # Simulator calls this ONCE per turn
    # =========================================================
    def step(self, env):
        # 1) Respect cooldown (same as all other agents)
        if self.cooldown > 0:
            self.cooldown -= 1
            return Actions.NO_OP, None

        # 2) Build minimax root state from environment
        #    NOTE: env already contains BOTH agents
        agent0, agent1 = env.agents
        root = build_initial_gamestate(env, agent0, agent1)

        # 3) Run minimax
        value, action = self._minimax(
            state=root,
            env=env,
            depth=0,
            alpha=float("-inf"),
            beta=float("inf"),
            visited=set()
        )

        # 4) Safety fallback
        if action is None:
            return Actions.NO_OP, None

        return action

    # =========================================================
    # Minimax with Alpha-Beta pruning
    # =========================================================
    def _minimax(self, state, env, depth, alpha, beta, visited):
        # Terminal or cutoff
        if self._is_terminal_or_cutoff(state, env, depth, visited):
            return self._evaluate(state), None

        visited.add(state.key())

        maximizing = (state.current_player() == self.id)

        best_action = None

        if maximizing:
            value = float("-inf")
            for next_state, action, _ in successors_game(state, env):
                v, _ = self._minimax(
                    next_state, env, depth + 1, alpha, beta, visited
                )
                if v > value:
                    value = v
                    best_action = action
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:
            value = float("inf")
            for next_state, action, _ in successors_game(state, env):
                v, _ = self._minimax(
                    next_state, env, depth + 1, alpha, beta, visited
                )
                if v < value:
                    value = v
                    best_action = action
                beta = min(beta, value)
                if beta <= alpha:
                    break

        visited.remove(state.key())
        return value, best_action

    # =========================================================
    # Terminal & Cutoff
    # =========================================================
    def _is_terminal_or_cutoff(self, state, env, depth, visited):
        if depth >= self.max_depth:
            return True

        if state.time >= env.deadline:
            return True

        if sum(state.remaining_people) == 0:
            return True

        if state.key() in visited:
            return True

        return False

    # =========================================================
    # Static Evaluation Function
    # =========================================================
    def _evaluate(self, state):
        s0, s1 = state.saved
        remaining = sum(state.remaining_people)

        # ---------- Adversarial ----------
        if self.game_type == "adversarial":
            val = (s0 - s1) if self.id == 0 else (s1 - s0)

        # ---------- Semi-cooperative ----------
        elif self.game_type == "semi":
            own = s0 if self.id == 0 else s1
            other = s1 if self.id == 0 else s0
            val = own + 0.001 * other  # cooperative tie-break

        # ---------- Fully cooperative ----------
        elif self.game_type == "cooperative":
            val = s0 + s1

        else:
            raise ValueError("Unknown game type")

        # Small heuristic push to finish saving people
        val -= 0.01 * remaining
        return val
