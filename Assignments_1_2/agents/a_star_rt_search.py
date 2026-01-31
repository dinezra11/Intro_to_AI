from Assignments_1_2.agents.base_agent import BaseAgent
from utils.constants import Actions
from utils.search import SearchState, successors
from Assignments_1_2.utils.heuristic import heuristic
import heapq
import itertools


class RealTimeAStar(BaseAgent):
    def __init__(self, id, initial_position, expansion_limit=3):
        super().__init__(id, initial_position)
        self.agent_type = 'A*-RealTime-Search'
        self.expansion_limit = expansion_limit  # 'L' parameter

    def step(self, env):
        # 1) Respect cooldown
        if self.cooldown > 0:
            self.cooldown -= 1
            return Actions.NO_OP, None

        # 2) RTA* does not store a plan. It plans one step at a time.

        # Build remaining_people vector
        n = env.n_vertices
        remaining_people = [0] * n
        for v, objs in enumerate(env.objects):
            for obj in objs:
                if obj.startswith('P'):
                    remaining_people[v] += int(obj[1:])

        if all(count == 0 for count in remaining_people):
            return Actions.NO_OP, None

        start_state = SearchState(
            position=self.position,
            remaining_people=tuple(remaining_people),
            has_kit=self.is_holding_amphibian,
            parent=None,
            action_from_parent=None
        )

        # Run Limited A* to decide NEXT action
        action, info = self.rta_star_search(start_state, env)
        return action, info

    def rta_star_search(self, start_state, env):
        open_list = []
        closed_g = {}
        counter = itertools.count()
        expansions = 0

        h0 = heuristic(start_state, env)
        # We use f = g + h.
        heapq.heappush(open_list, (h0, next(counter), start_state))
        closed_g[start_state.key()] = 0

        # Run the search loop limited by expansion_limit (L)
        while open_list and expansions < self.expansion_limit:
            f, _, state = heapq.heappop(open_list)
            expansions += 1

            # If we happen to find the goal within the limit, move towards it
            if all(count == 0 for count in state.remaining_people):
                return self.extract_next_action(state)

            current_g = closed_g[state.key()]

            for next_state, action_info, step_cost in successors(state, env):
                tentative_g = current_g + step_cost
                next_key = next_state.key()

                if next_key not in closed_g or tentative_g < closed_g[next_key]:
                    closed_g[next_key] = tentative_g
                    h = heuristic(next_state, env)
                    heapq.heappush(open_list, (tentative_g + h, next(counter), next_state))

        # Loop finished or limit reached.
        if not open_list:
            # No reachable states?
            return Actions.NO_OP, None

        # Identify the most promising node currently in the open_list (lowest f)
        # Since heapq is a min-heap, the first element is the best.
        best_f, _, best_state = open_list[0]

        # We move towards this 'best_state'
        return self.extract_next_action(best_state)

    def extract_next_action(self, target_state):
        """
        Backtrack from target_state up to the immediate child of the start_state.
        """
        curr = target_state

        # If target is start_state (e.g. limit=0 or no better options), we can't move 'to' it.
        if curr.parent is None:
            return Actions.NO_OP, None

        # Move up the tree until curr.parent is the root (start_state)
        while curr.parent.parent is not None:
            curr = curr.parent

        # curr is now the immediate child of start_state
        # curr.action_from_parent contains the action used to get here
        return curr.action_from_parent