from agents.base_agent import BaseAgent
from utils.constants import Actions
from utils.search import SearchState, successors  
from utils.heuristic import heuristic

import heapq
import itertools



class GreedySearch(BaseAgent):
    def __init__(self, id, initial_position):
        super().__init__(id, initial_position)
        self.agent_type = 'Greedy-Search'
        # This will hold a list of (action, info) pairs to execute step by step
        self._current_plan = []

    def step(self, env):
        """
        Called once per simulation tick.
        Must return (action, info) just like the other agents.
        """
        # 1) Respect cooldown (same as other agents)
        if self.cooldown > 0:
            self.cooldown -= 1
            return Actions.NO_OP, None

        # 2) If we already have a plan, keep following it
        if self._current_plan:
            action, info = self._current_plan.pop(0)
            return action, info

        # 3) Otherwise, we need to PLAN from scratch using greedy search

        # 3a) Build remaining_people vector from env
        #     remaining_people[v] = number of people still at vertex v
        n = env.n_vertices
        remaining_people = [0] * n
        for v, objs in enumerate(env.objects):
            for obj in objs:
                if obj.startswith('P'):
                    remaining_people[v] += int(obj[1:])

        # If no people left anywhere → do nothing
        if all(count == 0 for count in remaining_people):
            return Actions.NO_OP, None

        has_kit = self.is_holding_amphibian

        # 3b) Build the search start state
        start_state = SearchState(
            position=self.position,
            remaining_people=tuple(remaining_people),
            has_kit=has_kit,
            parent=None,
            action_from_parent=None
        )

        # 3c) Run greedy best-first search to get a full plan
        plan = self.greedy_search(start_state, env)

        # 3d) If search failed -> NO_OP
        if not plan:
            return Actions.NO_OP, None

        # 3e) Save plan and execute its first action
        self._current_plan = plan
        action, info = self._current_plan.pop(0)
        return action, info


    def greedy_search(self, start_state, env):
        """
        Greedy Best-First Search:
        - 'open_list' is ordered only by h(state)
        - 'visited' prevents re-expansion of identical states
        - returns a full action plan (list of (action, info))
        """
        open_list = []
        visited = set()

        # Counter creates a unique tie-breaker for heap entries.
        # This prevents comparing SearchState objects when two states have the same heuristic value.
        # Example heap entry: (h, tie_breaker, state)
        counter = itertools.count()

        # Push the start state
        h0 = heuristic(start_state, env)
        heapq.heappush(open_list, (h0, next(counter), start_state))

        while open_list:
            _, _, state = heapq.heappop(open_list)

            state_key = state.key()
            if state_key in visited:
                continue
            visited.add(state_key)

            # Goal test: no people left anywhere
            if all(count == 0 for count in state.remaining_people):
                return self.reconstruct_plan(state)

            # Expand successors
            for next_state, (action, info), _ in successors(state, env):
                next_state_key = next_state.key()
                if next_state_key in visited:
                    continue

                h = heuristic(next_state, env)
                heapq.heappush(open_list, (h, next(counter), next_state))

        # No plan found
        return []


    def reconstruct_plan(self, goal_state):
        """
        Walk backwards via parent pointers:
        goal_state -> ... -> start_state

        and collect the actions used at each step, then reverse them
        to get the correct order: start -> ... -> goal.

        Returns: list of (action, info) pairs.
        """
        actions = []
        state = goal_state

        while state.parent is not None:     # if parent is None -> this is the start state.
        
            actions.append(state.action_from_parent)   # action_from_parent is exactly the (action, info) that produced this state
            state = state.parent

        # We collected actions from goal → start, so reverse to get start → goal
        actions.reverse()
        return actions
