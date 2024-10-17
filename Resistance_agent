from agent import Agent
import random
from typing import List, Dict, Union

class MyAgent(Agent):
    # Static game configurations based on the number of players
    MISSION_SIZES = {
        5: [2, 3, 2, 3, 3],
        6: [2, 3, 4, 3, 4],
        7: [2, 3, 3, 4, 4],
        8: [3, 4, 4, 5, 5],
        9: [3, 4, 4, 5, 5],
        10: [3, 4, 4, 5, 5]
    }
    SPY_COUNT = {
        5: 2, 6: 2, 7: 3,
        8: 3, 9: 3, 10: 4
    }
    BETRAYALS_REQUIRED = {
        5: [1, 1, 1, 1, 1],
        6: [1, 1, 1, 1, 1],
        7: [1, 1, 1, 2, 1],
        8: [1, 1, 1, 2, 1],
        9: [1, 1, 1, 2, 1],
        10: [1, 1, 1, 2, 1]
    }

    def __init__(self, name: str):
        self.name = name

        # Core game state
        self.player_count = 0
        self.player_number = 0
        self.is_spy = False
        self.spies: List[int] = []
        self.rounds_completed = 0
        self.missions_failed = 0
        self.last_mission_team: List[int] = []

        # Player assessment data
        self.spy_probabilities: Dict[int, float] = {}
        self.trust_scores: Dict[int, float] = {}
        self.behavior_scores: Dict[int, float] = {}

        # Historical data
        self.vote_history: Dict[int, List[bool]] = {}
        self.mission_history: List[Dict] = []
        self.proposal_history: Dict[int, List[List[int]]] = {}

        # Reinforcement learning parameters
        self.q_values: Dict[tuple, Dict[Union[tuple, bool], float]] = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1  # Exploration rate for epsilon-greedy strategy
        
    def new_game(self, number_of_players: int, player_number: int, spies: List[int]):

        self.player_count = number_of_players
        self.player_number = player_number
        self.is_spy = player_number in spies
        self.spies = spies if self.is_spy else []

        # Initialize probability estimates and scores for each player
        self.spy_probabilities = {}
        self.trust_scores = {}
        self.behavior_scores = {}
        self.vote_history = {}
        self.proposal_history = {}
        self.q_values = {}
        self.rounds_completed = 0
        self.missions_failed = 0
        self.last_mission_team = []
        self.current_state = (self.rounds_completed, self.missions_failed)

        for i in range(number_of_players):
            if i != self.player_number:
                # Initial probability that a player is a spy
                self.spy_probabilities[i] = self.SPY_COUNT[number_of_players] / (number_of_players - 1)
            else:
                self.spy_probabilities[i] = 0.0  # The agent knows they are not a spy
            self.trust_scores[i] = 1.0  # Neutral trust score
            self.behavior_scores[i] = 0.5  # Neutral behavior score
            self.vote_history[i] = []  # Initialize empty vote history
            self.proposal_history[i] = []  # Initialize empty proposal history

    def get_action(self, state, possible_actions):

        random_value = random.random()
        if random_value < self.epsilon:
            # Explore - choose a random action
            chosen_action = random.choice(possible_actions)
        else:
            # Exploit - choose the best known action
            if state not in self.q_values:
                self.q_values[state] = {}
            state_q_values = self.q_values[state]

            if isinstance(possible_actions[0], list):
                # Actions are mission proposals
                max_q_value = None
                best_action = None
                for action_candidate in possible_actions:
                    # Convert action to a sorted tuple to use as a key
                    action_key = tuple(sorted(action_candidate))
                    # Get Q-value for this action, defaulting to 0 if not found
                    q_value = state_q_values.get(action_key, 0)
                    # Compare and keep the best action
                    if (max_q_value is None) or (q_value > max_q_value):
                        max_q_value = q_value
                        best_action = action_candidate
                chosen_action = best_action
            else:
                # Actions are boolean (True/False)
                max_q_value = None
                best_action = None
                for action_candidate in possible_actions:
                    # Use action_candidate as key
                    action_key = action_candidate
                    # Get Q-value for this action, defaulting to 0 if not found
                    q_value = state_q_values.get(action_key, 0)
                    # Compare and keep the best action
                    if (max_q_value is None) or (q_value > max_q_value):
                        max_q_value = q_value
                        best_action = action_candidate
                chosen_action = best_action

        return chosen_action

    def update_q_value(self, state, action, reward, next_state):

        if state not in self.q_values:
            self.q_values[state] = {}
        if next_state not in self.q_values:
            self.q_values[next_state] = {}

        # Prepare action key
        if isinstance(action, (tuple, list)):
            # Action is a list or tuple (mission proposal)
            action_key = tuple(sorted(action))
        else:
            # Action is a boolean value
            action_key = action

        current_q = self.q_values[state].get(action_key, 0)
        next_q_values = self.q_values[next_state]

        # Determine the maximum Q-value for the next state
        max_next_q = None
        if isinstance(action_key, tuple):
            # For mission proposals
            for next_action_key in next_q_values:
                if isinstance(next_action_key, tuple):
                    q_value = next_q_values.get(next_action_key, 0)
                    if (max_next_q is None) or (q_value > max_next_q):
                        max_next_q = q_value
            if max_next_q is None:
                max_next_q = 0
        else:
            # For boolean actions
            for next_action_key in next_q_values:
                if isinstance(next_action_key, bool):
                    q_value = next_q_values.get(next_action_key, 0)
                    if (max_next_q is None) or (q_value > max_next_q):
                        max_next_q = q_value
            if max_next_q is None:
                max_next_q = 0

        # Q-learning update formula
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_values[state][action_key] = new_q

    def propose_mission(self, team_size: int, betrayals_required: int = 1) -> List[int]:

        # Generate possible team proposals
        possible_teams = []
        if self.is_spy:
            for mission in range(5):
                team = self.spy_team_proposal(team_size, betrayals_required)
                possible_teams.append(team)
        else:
            for mission in range(5):
                team = self.resistance_team_proposal(team_size, betrayals_required)
                possible_teams.append(team)

        # Choose a team using the epsilon-greedy strategy
        chosen_team = self.get_action(self.current_state, possible_teams)

        # Update Q-value for the chosen team
        reward = 0  # Neutral reward at this point
        next_state = (self.rounds_completed, self.missions_failed)
        self.update_q_value(self.current_state, tuple(sorted(chosen_team)), reward, next_state)

        return chosen_team

    def spy_team_proposal(self, team_size: int, betrayals_required: int) -> List[int]:

        team = [self.player_number]

        # Include other spies if necessary
        other_spies = []
        for spy in self.spies:
            if spy != self.player_number:
                other_spies.append(spy)

        spies_needed = betrayals_required - 1
        spies_to_add = min(len(other_spies), spies_needed)
        if spies_to_add > 0:
            additional_spies = random.sample(other_spies, spies_to_add)
            team.extend(additional_spies)

        # Fill remaining slots with trusted Resistance members
        combined_scores = self.calculate_combined_scores()
        trusted_players = list(combined_scores.items())
        trusted_players.sort(key=lambda item: item[1], reverse=True)

        for player, score in trusted_players:
            if len(team) >= team_size:
                break
            if player not in team and player not in self.spies:
                team.append(player)

        return team

    def resistance_team_proposal(self, team_size: int, betrayals_required: int) -> List[int]:

        team = [self.player_number]  # Always include self
        combined_scores = self.calculate_combined_scores()

        # Include players from previous successful missions
        successful_players = []
        if self.mission_history and self.mission_history[-1]['success']:
            last_mission_team = self.mission_history[-1]['team']
            for player in last_mission_team:
                if player != self.player_number:
                    successful_players.append(player)

        for player in successful_players:
            if len(team) >= team_size:
                break
            if player not in team:
                team.append(player)   

        # Fill remaining slots based on trust scores
        trusted_players = list(combined_scores.items())
        trusted_players.sort(key=lambda item: item[1], reverse=True)

        for player, score in trusted_players:
            if len(team) >= team_size:
                break
            if player not in team:
                team.append(player)

        return team

    def vote(self, mission: List[int], proposer: int, betrayals_required: int = 1) -> bool:

        possible_votes = [True, False]
        rl_vote = self.get_action(self.current_state, possible_votes)

        if self.is_spy:
            # Spies may have additional considerations
            spy_vote = self.spy_vote(mission, betrayals_required)
            final_vote = rl_vote or spy_vote
        else:
            # Resistance members rely more on their assessment
            resistance_vote = self.resistance_vote(mission, proposer, betrayals_required)
            final_vote = rl_vote and resistance_vote

        # Update Q-value for the vote action
        reward = 0  # Neutral reward at this point
        next_state = (self.rounds_completed, self.missions_failed)
        self.update_q_value(self.current_state, final_vote, reward, next_state)

        return final_vote

    def spy_vote(self, mission: List[int], betrayals_required: int) -> bool:

        spies_on_mission = 0
        for player in mission:
            if player in self.spies:
                spies_on_mission += 1

        if spies_on_mission >= betrayals_required:
            return True  # Approve if sufficient spies are on the mission
        elif spies_on_mission == 0:
            # Small chance to approve to avoid suspicion
            approval_chance = 0.2
            random_value = random.random()
            return random_value < approval_chance
        else:
            # Adjust approval chance based on game context
            approval_chance = 0.35 + (0.1 * self.rounds_completed)
            if self.missions_failed >= 2:
                approval_chance += 0.2  # More likely to approve if spies are winning
            random_value = random.random()
            return random_value < approval_chance

    def resistance_vote(self, mission: List[int], proposer: int, betrayals_required: int) -> bool:

        # Calculate the expected number of spies on the mission
        team_spy_probability = 0
        for player in mission:
            if player != self.player_number:
                team_spy_probability += self.spy_probabilities.get(player, 0)

        expected_spies = team_spy_probability

        # Calculate dynamic threshold for approval
        threshold = self.calculate_voting_threshold(proposer, mission, betrayals_required)

        # Approve the mission if expected spies are below the threshold
        return expected_spies < threshold

    def calculate_voting_threshold(self, proposer: int, mission: List[int], betrayals_required: int) -> float:

        base_threshold = betrayals_required - 0.1
        round_adjustment = min(0.3, self.rounds_completed * 0.05)
        fail_adjustment = self.missions_failed * 0.1
        proposer_trust = (self.trust_scores.get(proposer, 1.0) - 1) * 0.1

        threshold = base_threshold + round_adjustment - fail_adjustment + proposer_trust

        # Additional adjustments based on game state
        if self.rounds_completed < 2:
            threshold -= 0.2  # More skeptical in early rounds
            if self.player_number not in mission:
                threshold -= 0.15  # More skeptical if not on the mission
        elif self.rounds_completed >= 3:
            if self.missions_failed >= 2:
                threshold += 0.25  # More lenient if losing
            elif self.missions_failed == 0:
                threshold -= 0.15  # More strict if winning

        return threshold

    def betray(self, mission: List[int], proposer: int, betrayals_required: int = 1) -> bool:

        if not self.is_spy:
            return False  # Resistance members cannot betray

        # Decide whether to betray using epsilon-greedy strategy
        possible_actions = [True, False]
        action = self.get_action(self.current_state, possible_actions)

        # Update Q-value for the betrayal action
        reward = 0  # Neutral reward at this point
        next_state = (self.rounds_completed, self.missions_failed)
        self.update_q_value(self.current_state, action, reward, next_state)

        return action

    def vote_outcome(self, mission: List[int], proposer: int, votes: Union[List[bool], Dict[int, bool]]):

        # Process votes and update vote history
        if isinstance(votes, dict):
            vote_items = votes.items()
        else:
            vote_items = list(enumerate(votes))

        for player, vote in vote_items:
            if player != self.player_number:
                if player not in self.vote_history:
                    self.vote_history[player] = []
                self.vote_history[player].append(vote)

        # Record the proposal
        if proposer not in self.proposal_history:
            self.proposal_history[proposer] = []
        self.proposal_history[proposer].append(mission)

        # Update behavior scores
        self.update_behavior_scores(mission, proposer, votes)

    def mission_outcome(self, mission: List[int], proposer: int, num_betrayals: int, mission_success: bool):

        # Record mission details
        mission_data = {
            'team': mission,
            'proposer': proposer,
            'betrayals': num_betrayals,
            'success': mission_success
        }
        self.mission_history.append(mission_data)
        self.last_mission_team = mission

        # Update spy probabilities and trust scores
        if not self.is_spy:
            self.update_spy_probabilities(mission, num_betrayals, mission_success)
        self.update_trust_scores(mission, num_betrayals, mission_success)

        # Update Q-value for mission outcome
        reward = 1 if mission_success else -1
        next_state = (self.rounds_completed, self.missions_failed)
        self.update_q_value(self.current_state, mission, reward, next_state)
        self.current_state = next_state  # Update current state

    def update_spy_probabilities(self, mission: List[int], num_betrayals: int, mission_success: bool):

        team_size = len(mission)
        round_factor = min(2, 1 + self.rounds_completed / 5)

        for player in self.spy_probabilities:
            if player in mission:
                if mission_success:
                    # Decrease spy probability for team members if mission succeeded
                    adjustment_factor = (0.9 ** (team_size - num_betrayals) / team_size) ** round_factor
                    self.spy_probabilities[player] *= adjustment_factor
                else:
                    # Increase spy probability for team members if mission failed
                    adjustment_factor = (1.1 ** num_betrayals / team_size) ** round_factor
                    self.spy_probabilities[player] *= adjustment_factor
            else:
                # Adjust spy probability for players not on mission
                if mission_success:
                    self.spy_probabilities[player] *= (1.02) ** round_factor
                else:
                    self.spy_probabilities[player] *= (0.98) ** round_factor

        # Normalize probabilities
        total_probability = sum(self.spy_probabilities.values())
        if total_probability > 0:
            for player in self.spy_probabilities:
                self.spy_probabilities[player] /= total_probability

    def update_trust_scores(self, mission: List[int], num_betrayals: int, mission_success: bool):
 
        for player in mission:
            if player != self.player_number:
                if mission_success:
                    # Increase trust for team members if mission succeeded
                    self.trust_scores[player] *= 1.1
                else:
                    # Decrease trust based on betrayal probability
                    betrayal_probability = num_betrayals / len(mission)
                    self.trust_scores[player] *= (1 - betrayal_probability * 0.5)

    def update_behavior_scores(self, mission: List[int], proposer: int, votes: Union[List[bool], Dict[int, bool]]):

        if isinstance(votes, dict):
            vote_items = votes.items()
        else:
            vote_items = list(enumerate(votes))

        for player, vote in vote_items:
            if player != self.player_number:
                expected_vote = self.predict_vote(player, mission, proposer)
                if vote == expected_vote:
                    # Increase behavior score if vote was as expected
                    self.behavior_scores[player] += 0.05
                else:
                    # Decrease behavior score if vote was unexpected
                    self.behavior_scores[player] -= 0.1
                # Clamp behavior scores between 0 and 1
                self.behavior_scores[player] = max(0, min(1, self.behavior_scores[player]))

    def predict_vote(self, player: int, mission: List[int], proposer: int) -> bool:

        if player in mission:
            return True  # Players on the mission are expected to approve

        # Calculate expected number of spies on the mission
        team_spy_probability = 0
        for team_member in mission:
            if team_member != player:
                team_spy_probability += self.spy_probabilities.get(team_member, 0)

        mission_size = len(mission)
        if mission_size == 0:
            return False  # Avoid division by zero

        # Expect approval if expected spies are less than half the team size
        return team_spy_probability < (mission_size / 2)

    def calculate_combined_scores(self) -> Dict[int, float]:

        #Calculate combined scores for players based on trust, spy probability, and behavior.

        combined_scores = {}
        for player in self.trust_scores:
            trust_component = self.trust_scores[player] * 0.55
            spy_probability = self.spy_probabilities.get(player, 0)
            spy_component = (1 - spy_probability) * 0.3
            behavior_component = self.behavior_scores.get(player, 0.5) * 0.15
            combined_score = trust_component + spy_component + behavior_component
            combined_scores[player] = combined_score
        return combined_scores

    def round_outcome(self, rounds_complete: int, missions_failed: int):

        self.rounds_completed = rounds_complete
        self.missions_failed = missions_failed

    def game_outcome(self, spies_win: bool, spies: List[int]):

        # Calculate reward based on game outcome
        if (self.is_spy and spies_win) or (not self.is_spy and not spies_win):
            reward = 1  # Agent's team won
        else:
            reward = -1  # Agent's team lost

        # Update Q-value for game end
        self.update_q_value(self.current_state, 'game_end', reward, (0, 0))

        # Reduce epsilon over time to favor exploitation over exploration
        self.epsilon = max(0.01, self.epsilon * 0.99)
