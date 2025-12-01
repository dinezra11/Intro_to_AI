from agents.base_agent import BaseAgent
from utils.constants import Actions
from utils.search import SearchState, successors
from utils.heuristic import heuristic
import heapq
import itertools

class AStarSearch(BaseAgent):
    def __init__(self, id, initial_position):
        super().__init__(id, initial_position)
        self.agent_type = 'A*-Search'
        self.limit = 10000  # Global limit for expansions as per assignment
        self._current_plan = []

    def step(self, env):
        # 1) Respect cooldown
        if self.cooldown > 0:
            self.cooldown -= 1
            return Actions.NO_OP, None

        # 2) If we already have a plan, keep following it
        if self._current_plan:
            action, info = self._current_plan.pop(0)
            return action, info

        # 3) Otherwise, PLAN from scratch using A* Search

        # Build remaining_people vector from env
        n = env.n_vertices
        remaining_people = [0] * n
        for v, objs in enumerate(env.objects):
            for obj in objs:
                if obj.startswith('P'):
                    remaining_people[v] += int(obj[1:])

        # If no people left anywhere -> do nothing
        if all(count == 0 for count in remaining_people):
            return Actions.NO_OP, None

        start_state = SearchState(
            position=self.position,
            remaining_people=tuple(remaining_people),
            has_kit=self.is_holding_amphibian,
            parent=None,
            action_from_parent=None
        )

        # Run A* search
        plan = self.a_star_search(start_state, env)

        # If search failed (e.g. limit reached) -> NO_OP (Terminate)
        if not plan:
            return Actions.NO_OP, None

        # Save plan and execute its first action
        self._current_plan = plan
        action, info = self._current_plan.pop(0)
        return action, info

    def a_star_search(self, start_state, env):
        open_list = []
        closed_g = {} # Maps state key -> lowest g_score found so far
        counter = itertools.count() # Unique tie-breaker
        expansions = 0

        # f = g + h. Initially g=0.
        h0 = heuristic(start_state, env)
        heapq.heappush(open_list, (h0, next(counter), start_state))
        closed_g[start_state.key()] = 0

        while open_list:
            if expansions >= self.limit:
                return [] # Failed: Limit reached

            f, _, state = heapq.heappop(open_list)
            expansions += 1

            # Goal test
            if all(count == 0 for count in state.remaining_people):
                return self.reconstruct_plan(state)

            current_g = closed_g[state.key()]

            # Expand successors
            for next_state, action_info, step_cost in successors(state, env):
                tentative_g = current_g + step_cost
                next_key = next_state.key()

                # If this is a better path to next_state, record it and push to open list
                if next_key not in closed_g or tentative_g < closed_g[next_key]:
                    closed_g[next_key] = tentative_g
                    h = heuristic(next_state, env)
                    f_new = tentative_g + h
                    heapq.heappush(open_list, (f_new, next(counter), next_state))

        return [] # No solution found

    def reconstruct_plan(self, goal_state):
        actions = []
        state = goal_state
        while state.parent is not None:
            actions.append(state.action_from_parent)
            state = state.parent
        actions.reverse()
        return actions