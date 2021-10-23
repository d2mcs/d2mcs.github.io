"""This module contains code for computing TI10 group stage
probabilities using Monte-Carlo sampling. Multiprocessing is used to
speed up computation of results.
"""
from multiprocessing import Pool
import random
import copy
import json

from tqdm import tqdm

from model.forecaster import PlayerModel, TeamModel
from model.forecaster_glicko import Glicko2Model
from model.tiebreakers import Tiebreaker

class Simulator:
    """Generic simulator class to be subclassed by more specific
    simulator code. Contains utilities for loading a model/rosters and
    simulating best of n matches.

    Parameters
    ----------
    model : TeamModel or PlayerModel
        Elo model to compute win probabilities and team ratings.
    model_type : ["team", "player"]
        Indicates the type of model used for the simulation.
    rosters : dict
        Dict mapping team names to player IDs. Example:
        {
          "Team A": [1, 2, 3, 4, 5],
          "Team B": [6, 7, 8, 9, 10]
        }
        For a TeamModel it's fine for the rosters to be empty lists,
        but the team names still must be present.
    static_ratings : bool, default=False
        If False, team ratings will be updated over the course of the
        simulation based on the simulated results (e.g., if Team A is
        simulated to have won a match their rating will be updated
        accordingly). The wider resulting distribution will be a little
        less prone to excessive confidence, which is particularly
        helpful for making sure an output probability of 0 or 100 is
        actually impossible/guaranteed.

        Setting this option to True results in a tighter distribution
        and a significantly faster runtime.
    """
    def __init__(self, model, model_type, rosters, static_ratings=False):
        self.model = model
        self.model_type = model_type
        self.rosters = rosters
        self.static_ratings = static_ratings

    @classmethod
    def from_ratings_file(cls, ratings_file, k, static_ratings=False):
        """Constructs and returns a Simulator object from a file
        containing precomputed / manually determined team ratings.

        Parameters
        ----------
        ratings_file : str
            Path to JSON file mapping team names to ratings. The file
            should look something like this:
            {
              "Team A": 1500,
              "Team B": 1600
            }
        k : int
            k parameter for player model. See model documentation for
            details
        static_ratings : bool
            (See: constructor documentation above)

        Returns
        -------
        Simulator
            Constructed simulator object
        """
        with open(ratings_file) as rating_f:
            ratings = json.load(rating_f)
        model = TeamModel(ratings, k)
        rosters = {team: [] for team in ratings}
        return cls(model, "team", rosters, static_ratings=static_ratings)

    @classmethod
    def from_ratings_file_glicko2(cls, ratings_file, tau,static_ratings=False):
        """Constructs and returns a Simulator object from a file
        containing precomputed / manually determined team ratings.

        Parameters
        ----------
        ratings_file : str
            Path to JSON file mapping team names to ratings in the form
            of a 3 tuple containing (rating, RD, volatility). The file
            should look something like this:
            {
              "Team A": [1500, 350, 0.6],
              "Team B": [1600, 300, 0.59]
            }
            Ratings/RDs should use the Glicko-1 scale rather than the
            Glicko-2 scale (e.g., default ratings should be 1500).
        tau : int
            tau parameter for player model. See model documentation for
            details
        static_ratings : bool
            (See: constructor documentation above)

        Returns
        -------
        Simulator
            Constructed simulator object
        """
        with open(ratings_file) as rating_f:
            ratings = json.load(rating_f)
        model = Glicko2Model(tau)
        for team, rating_tuple in ratings.items():
            model.team_ratings[team] = ((rating_tuple[0] - 1500)/173.7178,
                rating_tuple[1]/173.7178, rating_tuple[2])
        rosters = {team: [] for team in ratings}
        return cls(model, "team", rosters, static_ratings=static_ratings)

    def _get_team(self, team_name):
        """Private team name wrapper function to ensure identical code
        regardless of whether the model is player-based or team based
        """
        if self.model_type == "player":
            return self.rosters[team_name]
        else:
            return team_name

    def sim_bo1(self, team1, team2):
        """Simulates a best-of-1 match.

        Parameters
        ----------
        team1 : str or list of int
            Team name (TeamModel) or list of player IDs (PlayerModel).
        team2 : str or list of int
            Team name (TeamModel) or list of player IDs (PlayerModel).

        Returns
        -------
        bool
            True if team 1 wins, False if team 2 wins
        """
        _team1 = self._get_team(team1)
        _team2 = self._get_team(team2)

        win_p_t1 = self.model.get_win_prob(_team1, _team2)
        team1_win = random.random() < win_p_t1
        if not self.static_ratings:
            self.model.update_ratings(_team1, _team2, (team1_win, 1-team1_win))
        return team1_win

    def sim_bo2(self, team1, team2, momentum=0.0):
        """Simulates a best-of-2 match.

        Parameters
        ----------
        team1 : str or list of int
            Team name (TeamModel) or list of player IDs (PlayerModel).
        team2 : str or list of int
            Team name (TeamModel) or list of player IDs (PlayerModel).
        momentum : float, default=0.0
            Using game probabilities alone results in consistent over-
            estimation of draw probabilities in bo2s. Momentum adds a
            given percentage to the win probability of the team that
            won game one to reduce draw probability.
        Returns
        -------
        tuple of (int, int)
            Number of wins for (team 1, team 2)
        """
        _team1 = self._get_team(team1)
        _team2 = self._get_team(team2)

        team1_wins = 0
        team2_wins = 0
        win_p_t1 = self.model.get_win_prob(_team1, team2)
        for _ in range(2):
            team1_win = random.random() < win_p_t1
            team1_wins += int(team1_win)
            team2_wins += 1 - int(team1_win)
            if momentum > 0:
                if team1_win:
                    win_p_t1 = min(max(0.95, win_p_t1), win_p_t1 + momentum)
                else:
                    win_p_t1 = max(min(0.05, win_p_t1), win_p_t1 - momentum)
        if not self.static_ratings:
            self.model.update_ratings(_team1, team2, (team1_wins, team2_wins))
        return (team1_wins, team2_wins)

    def sim_bo_n(self, n, team1, team2):
        """Simulates a best-of-n match, where n is odd.

        Parameters
        ----------
        n : int
            Number of games a team must with the majority of to win
            the match.
        team1 : str or list of int
            Team name (TeamModel) or list of player IDs (PlayerModel).
        team2 : str or list of int
            Team name (TeamModel) or list of player IDs (PlayerModel).

        Returns
        -------
        tuple of (int, int)
            Number of wins for (team 1, team 2)
        """
        if n % 2 != 1:
            raise ValueError("n must be odd")
        _team1 = self._get_team(team1)
        _team2 = self._get_team(team2)

        team1_wins = 0
        team2_wins = 0
        win_p_t1 = self.model.get_win_prob(_team1, _team2)
        while team1_wins < n/2 and team2_wins < n/2:
            team1_win = random.random() < win_p_t1
            team1_wins += int(team1_win)
            team2_wins += 1 - int(team1_win)
        if not self.static_ratings:
            self.model.update_ratings(_team1, _team2, (team1_wins, team2_wins))
        return (team1_wins, team2_wins)

    def get_bo2_probs(self, team1, team2, draw_adjustment=False):
        """Computes win/draw/loss probabilities for a bo2 match.

        Parameters
        ----------
        team1 : list of int
            list of player IDs for team 1.
        team2 : list of int
            list of player IDs for team 2.
        draw_adjustment : bool, default=False
            Using game probabilities alone results in consistent over-
            estimation of draw probabilities in bo2s. This counteracts
            that error by reducing draw change by 10% for most matches.

        Returns
        -------
        tuple of float
            Probability of 2-0, 1-1, and 0-2 results (in that order).
            Should always sum to 1
        """
        win_p_t1 = self.model.get_win_prob(self._get_team(team1),
                                           self._get_team(team2))
        if not draw_adjustment:
            p2_0 = pow(win_p_t1, 2)
            p0_2 = pow(1 - win_p_t1, 2)
        else:
            # winner probability adjustment shouldn't exceed 95%
            p2_0 = win_p_t1*min(max(0.95, win_p_t1), win_p_t1 + 0.1)
            p0_2 = (1-win_p_t1)*min(max(0.95, 1-win_p_t1), (1-win_p_t1) + 0.1)
        return (p2_0, (1 - (p2_0 + p0_2)), p0_2)

    def save_ratings(self, output_file):
        """Saves model ratings to a JSON file (nicely formatted for
        readability). Can be loaded by from_ratings_file.

        Parameters
        ----------
        output_file : str
            File path to write to.
        """
        with open(output_file, "w") as output_f:
            output_f.write("{\n")
            for i, team in enumerate(self.rosters.keys()):
                rating = self.model.get_team_rating(self._get_team(team))
                if i != len(self.rosters) - 1:
                    output_f.write(f'  "{team}": {rating:.2f},\n')
                else:
                    output_f.write(f'  "{team}": {rating:.2f}\n')
            output_f.write("}\n")

class TIEliminationBracket(Simulator):
    """16-team double elimination bracket Simulator. Like GroupStage,
    this class is currently designed specifically for the TI format
    (the seeding code assumes two groups of 8).

    Parameters
    ----------
    Identical to Simulator. See above for details
    """
    def seed(self, seeds):
        """Seeds the bracket using an ordered list of teams for each
        group.

        Parameters
        ----------
        seeds : dict
            Ordered list of of team names for each group. Should be in
            the following format:
            {
                "a" : ["Team A", "Team B"],
                "b" : ["Team C", "Team D"],
            }
        """
        # top seeds pick their opponents at random
        picks = random.choices([0,1], k=4)
        self.bracket = {
            "UB-R1": [
                [seeds["a"][0], seeds["b"][2 + picks[0]], (0, 0)],
                [seeds["b"][1], seeds["a"][2 + (1 - picks[1])], (0, 0)],
                [seeds["b"][0], seeds["a"][2 + picks[1]], (0, 0)],
                [seeds["a"][1], seeds["b"][2 + (1 - picks[0])], (0, 0)],
            ],
            "UB-R2": [[None, None, (0, 0)], [None, None, (0, 0)]],
            "UB-F": [[None, None, (0, 0)]],
            "LB-R1": [
                [seeds["a"][4], seeds["b"][6 + picks[2]], (0, 0)],
                [seeds["b"][5], seeds["a"][6 + (1 - picks[3])], (0, 0)],
                [seeds["b"][4], seeds["a"][6 + picks[3]], (0, 0)],
                [seeds["a"][5], seeds["b"][6 + (1 - picks[2])], (0, 0)],
            ],
            "LB-R2": [[None, None, (0, 0)], [None, None, (0, 0)],
                      [None, None, (0, 0)], [None, None, (0, 0)]],
            "LB-R3": [[None, None, (0, 0)], [None, None, (0, 0)]],
            "LB-R4": [[None, None, (0, 0)], [None, None, (0, 0)]],
            "LB-R5": [[None, None, (0, 0)]],
            "LB-F": [[None, None, (0, 0)]],
            "GF": [[None, None, (0, 0)]]
        }

    def load_bracket(self, bracket):
        self.bracket = bracket

    def _get_winner(self, round, n, i):
        """Private helper function which gets the winner of a match
        from the bracket if the match has happens and simulates it
        otherwise.
        """
        match = self.bracket[round][i]
        if max(match[2]) < n//2 + 1:
            result = self.sim_bo_n(n, match[0], match[1])
            match[2] = result
        elif not self.static_ratings:
            self.model.update_ratings(*match)
        return 1 - int(match[2][0] == n//2 + 1)

    def simulate(self):
        """Simulates every game in the bracket and returns the
        resulting placement of each team.

        Returns
        -------
        dict
            dictionary mapping placement (in the form of a string) to
            teams which received that placement. For consistency all
            dict values are sets even for ranks that only can contain
            one team
        """
        ranks = {"13-16": set(), "9-12": set(), "7-8": set(), "5-6": set(),
                 "4": set(), "3": set(), "2": set(), "1": set()}
        for i, match in enumerate(self.bracket["UB-R1"]):
            winner = self._get_winner("UB-R1", 3, i)
            self.bracket["UB-R2"][i//2][i % 2] = match[winner]
            self.bracket["LB-R2"][i][0] = match[1 - winner]
        for i, match in enumerate(self.bracket["LB-R1"]):
            winner = self._get_winner("LB-R1", 1, i)
            self.bracket["LB-R2"][i][1] = match[winner]
            ranks["13-16"].add(match[1 - winner])
        for i, match in enumerate(self.bracket["LB-R2"]):
            winner = self._get_winner("LB-R2", 3, i)
            self.bracket["LB-R3"][i//2][i % 2] = match[winner]
            ranks["9-12"].add(match[1 - winner])
        for i, match in enumerate(self.bracket["UB-R2"]):
            winner = self._get_winner("UB-R2", 3, i)
            self.bracket["UB-F"][0][i] = match[winner]
            self.bracket["LB-R4"][1 - i][0] = match[1 - winner]
        for i, match in enumerate(self.bracket["LB-R3"]):
            winner = self._get_winner("LB-R3", 3, i)
            self.bracket["LB-R4"][i][1] = match[winner]
            ranks["7-8"].add(match[1 - winner])
        for i, match in enumerate(self.bracket["LB-R4"]):
            winner = self._get_winner("LB-R4", 3, i)
            self.bracket["LB-R5"][0][i] = match[winner]
            ranks["5-6"].add(match[1 - winner])
        match = self.bracket["UB-F"][0]
        winner = self._get_winner("UB-F", 3, 0)
        self.bracket["GF"][0][0] = match[winner]
        self.bracket["LB-F"][0][0] = match[1 - winner]

        match = self.bracket["LB-R5"][0]
        winner = self._get_winner("LB-R5", 3, 0)
        self.bracket["LB-F"][0][1] = match[winner]
        ranks["4"].add(match[1 - winner])

        match = self.bracket["LB-F"][0]
        winner = self._get_winner("LB-F", 3, 0)
        self.bracket["GF"][0][1] = match[winner]
        ranks["3"].add(match[1 - winner])

        match = self.bracket["GF"][0]
        winner = self._get_winner("GF", 5, 0)
        ranks["2"].add(match[1 - winner])
        ranks["1"].add(match[winner])
        return ranks

class TIGroupStage(Simulator):
    """TI Group Stage Simulator: 9 team round-robin with additional
    matches played for ties along 4th-5th and 8th-9th place, head-to-
    head results used to break other ties.

    Parameters
    ----------
    Identical to Simulator. See above for details
    """

    def simulate(self, group, matches):
        """Simulates a single group stage and returns the resulting
        team ranks.

        Parameters
        ----------
        group : list of str
            list of team names in the group
        matches : list
            List of matches. Each match is a 3-element list containing
            team 1, team 2, and the match result as an int (0 for a
            0-2, 1 for a 1-1, 2 for a 2-0, and -1 if the match hasn't
            happened yet). Example:

                [["Team A", "Team B", 0], ["Team B", "Team C", 1],
                 ["Team D", "Team E", 2], ["Team E", "Team D", -1]]

            In this case the results are A 0-2 B, B 1-1 C, D 2-0 E, and
            E vs D has not yet been played.

        Returns
        -------
        list of list(str, int)
            Sorted list of teams and points. Example:
                [["Team A", 10], ["Team B", 9], ["Team C", 5]]
        dict
            dict mapping each boundary to the number of teams tied
            along that boundary
        """
        records = {team: [0,0,0] for team in group}
        h2h_results = {team: {} for team in group}
        tiebreak_sizes = []
        points = []

        all_matches = []
        base_k = self.model.k
        for match_day, match_day_list in enumerate(matches):
            self.model.k = base_k + base_k*(3 - match_day)/3
            for match in match_day_list:
                if match[2] == -1:
                    if self.static_ratings:
                        result = self.sim_bo2(match[0], match[1])
                    else:
                        result = self.sim_bo2(match[0], match[1], momentum=0.1)
                else:
                    result = (match[2], 2 - match[2])
                    if not self.static_ratings:
                        # use normal k parameter for actual results
                        self.model.k = base_k
                        self.model.update_ratings(self._get_team(match[0]),
                            self._get_team(match[1]), result)
                        self.model.k = base_k + base_k*(3 - match_day)/3
                records[match[0]][2 - result[0]] += 1
                records[match[1]][2 - result[1]] += 1
                h2h_results[match[0]][match[1]] = result[0]
                h2h_results[match[1]][match[0]] = result[1]

        point_map = {}
        for team, record in records.items():
            point_map[team] = record[0]*2 + record[1]

        tiebreaker = Tiebreaker(self)
        team_order = tiebreaker.order_teams(point_map.keys(), point_map)

        team_order, tiebreak_sizes = tiebreaker.boundary_tiebreak(
            [(3,4), (7,8)], team_order, point_map)
        team_order = tiebreaker.h2h_tiebreak(h2h_results,
                                             team_order, point_map)
        points = [(team, point_map[team]) for team in team_order]

        return points, tiebreak_sizes

class TISimulator(Simulator):
    """TI Simulator. For each Monte Carlo sample a GroupStage object
    and an EliminationBracket object are created using a working copy
    of the prediction model. Multiprocessing is then used to collect
    many samples reasonably quickly.

    Parameters
    ----------
    Identical to Simulator. See above for details
    """
    def get_sample(self, groups, matches):
        """Gets a single sample of group stage placements and final
        ranks using the GroupStage and EliminationBracket simulators.

        Parameters
        ----------
        groups : dict
            (See: sim_group_stage documentation below)
        matches : dict
            (See: sim_group_stage documentation below)

        Returns
        -------
            dict
                Ordered (team, points) tuple for each group.
            dict
                Size of tiebreak required along each boundary. 0 if no
                tiebreaker matches were necessary.
            dict
                Mapping from ranks to the teams placed at that rank.
        """
        if not self.static_ratings:
            # if ratings are updated during a simulation, the code
            # needs to work with a copy to make sure each simulation
            # is fresh
            working_model = copy.deepcopy(self.model)
        else:
            working_model = self.model
        gs_sim = TIGroupStage(working_model, "team",
                            self.rosters, self.static_ratings)

        points_a, tiebreak_sizes_a = gs_sim.simulate(groups["a"], matches["a"])
        points_b, tiebreak_sizes_b = gs_sim.simulate(groups["b"], matches["b"])

        bracket_sim = TIEliminationBracket(gs_sim.model, "team",
                                           self.rosters, self.static_ratings)
        bracket_sim.seed({"a": [p[0] for p in points_a],
                          "b": [p[0] for p in points_b]})
        ranks = bracket_sim.simulate()

        return ({"a": points_a, "b": points_b},
                {"a": tiebreak_sizes_a, "b": tiebreak_sizes_b},
                ranks)

    def get_sample_main_event(self, groups, matches, bracket):
        """Gets a single sample of final ranks from a single
        EliminationBracket simulation.

        Parameters
        ----------
        groups : dict
            (See: sim_group_stage documentation below)
        matches : dict
            (See: sim_group_stage documentation below)
        bracket : dict
            Bracket with team seeds and match results (results
            currently are not used)

        Returns
        -------
            dict
                Mapping from ranks to the teams placed at that rank
        """
        if not self.static_ratings:
            working_model = copy.deepcopy(self.model)
        else:
            working_model = self.model

        bracket_sim = TIEliminationBracket(working_model, "team",
                                           self.rosters, self.static_ratings)
        bracket_sim.load_bracket(bracket)
        ranks = bracket_sim.simulate()

        return ranks

    def sim_group_stage(self, groups, matches, n_trials):
        """Wrapper for running get_sample many many times. Uses
        multiprocessing to speed up results, which it combines
        afterwards to construct the output distribution.

        Parameters
        ----------
        groups : dict
            Mapping between groups (assumed to be 'a' and 'b') and the
            teams in those groups. Example:

                {"a": ["Team A", "Team B"], "b": ["Team C", "Team D"]}

        matches : dict
            List of matches for each group. Each match is a 3-element
            list containing team 1, team 2, and the match result as an
            int (0 for a 0-2, 1 for a 1-1, 2 for a 2-0, and -1 if the
            match hasn't happened yet). Example:
            {
                "a": [["Team A", "Team B", 0], ["Team B", "Team C", 1]],
                "b": [["Team D", "Team E", 2], ["Team E", "Team D", -1]]
            }
            In this case the results are A 0-2 B, B 1-1 C, D 2-0 E, and
            E vs D has not yet been played.
        n_trials : int
            Number of simulations to run.

        Returns
        -------
        dict
            List of probabilities for each rank for each team in each
            group
        """
        group_rank_probs = {
            "a": {team: [0 for _ in range(len(groups["a"]))]
                  for team in groups["a"]},
            "b": {team: [0 for _ in range(len(groups["b"]))]
                  for team in groups["b"]}}
        tiebreak_probs = {
            "a": {boundary: [0 for _ in range(len(groups["a"]) - 1)]
                  for boundary in [3,7]},
            "b": {boundary: [0 for _ in range(len(groups["b"]) - 1)]
                  for boundary in [3,7]}}
        final_rank_probs = {team: {
            "17-18": 0, "13-16": 0, "9-12": 0, "7-8": 0, "5-6": 0,
            "4": 0, "3": 0, "2": 0, "1": 0} for team in self.rosters.keys()
        }
        point_rank_probs = {
            group: {
                team: {
                    points: {
                        rank: 0 for rank in range(len(groups[group]))
                    } for points in range(len(groups[group])*2 - 1)
                } for team in groups[group]
            } for group in ["a", "b"] }
        record_probs = {group: {team: [0 for _ in range(len(groups["a"])*2-1)]
                        for team in groups[group]} for group in ["a", "b"]}

        # all results are stored in memory until the pool completes,
        # so pool size is limited to 1,000 to reduce memory usage
        remaining_trials = n_trials
        with tqdm(total=n_trials) as pbar:
            while remaining_trials > 0:
                pool_size = min(1000, remaining_trials)
                remaining_trials -= pool_size

                pool = Pool()
                sim_results = [pool.apply_async(self.get_sample, (
                    groups, matches)) for _ in range(pool_size)]
                for sim_result in sim_results:
                    points, tiebreak_sizes, ranks = sim_result.get()
                    for group in ["a", "b"]:
                        for i, (team, record) in enumerate(points[group]):
                            group_rank_probs[group][team][i] += 1/n_trials
                            if i == 8:
                                final_rank_probs[team]["17-18"] += 1/n_trials
                            point_rank_probs[group][team][record][i] += 1
                            record_probs[group][team][record] += 1/n_trials
                        for boundary, size in tiebreak_sizes[group].items():
                            tiebreak_probs[group][boundary][
                                size - 2] += 1/n_trials
                    for rank, teams in ranks.items():
                        for team in teams:
                            final_rank_probs[team][rank] += 1/n_trials
                pbar.update(pool_size)

        for group in ["a", "b"]:
            for team, record_map in point_rank_probs[group].items():
                for record, point_counts in record_map.items():
                    points_sum = sum(point_counts.values())
                    for points, amount in point_counts.items():
                        if amount > 0:
                            point_counts[points] = amount / points_sum
        return (group_rank_probs, tiebreak_probs,
                final_rank_probs, record_probs, point_rank_probs)

    def sim_main_event(self, groups, matches, bracket, n_trials):
        """Wrapper for running get_sample_main_event many times.
        Arguments and return values are identical to sim_group_stage
        with the exception of bracket, which should contain bracket
        seeding and results (see main_event_matches.json in ti10/data
        for an example)
        """
        group_rank_probs = {
            "a": {team: [0 for _ in range(len(groups["a"]))]
                  for team in groups["a"]},
            "b": {team: [0 for _ in range(len(groups["b"]))]
                  for team in groups["b"]}}
        tiebreak_probs = {
            "a": {boundary: [0 for _ in range(len(groups["a"]) - 1)]
                  for boundary in [3,7]},
            "b": {boundary: [0 for _ in range(len(groups["b"]) - 1)]
                  for boundary in [3,7]}}
        final_rank_probs = {team: {
            "17-18": 0, "13-16": 0, "9-12": 0, "7-8": 0, "5-6": 0,
            "4": 0, "3": 0, "2": 0, "1": 0} for team in self.rosters.keys()
        }
        point_rank_probs = {
            group: {
                team: {
                    points: {
                        rank: 0 for rank in range(len(groups[group]))
                    } for points in range(len(groups[group])*2 - 1)
                } for team in groups[group]
            } for group in ["a", "b"] }
        record_probs = {group: {team: [0 for _ in range(len(groups["a"])*2-1)]
                        for team in groups[group]} for group in ["a", "b"]}

        # easiest way to get correct group stage data is just to use
        # the simulator code with a single sample
        points, tiebreak_sizes, ranks = self.get_sample(groups, matches)
        for group in ["a", "b"]:
            for i, (team, record) in enumerate(points[group]):
                group_rank_probs[group][team][i] += 1
                if i == 8:
                    final_rank_probs[team]["17-18"] += 1
                point_rank_probs[group][team][record][i] += 1
                record_probs[group][team][record] += 1
            for boundary, size in tiebreak_sizes[group].items():
                tiebreak_probs[group][boundary][size - 2] += 1

        # get_sample doesn't update team ratings, so that has to
        # happen seperately. sim_group_stage and sim_main_event aren't
        # supposed to update the model so a copy has to be saved so it
        # can be restored afterwards
        if not self.static_ratings:
            model = copy.deepcopy(self.model)
            for group in ["a", "b"]:
                for match_list in matches[group]:
                    for match in match_list:
                        result = (match[2], 2 - match[2])
                        self.model.update_ratings(match[0], match[1], result)

        remaining_trials = n_trials
        with tqdm(total=n_trials) as pbar:
            while remaining_trials > 0:
                pool_size = min(1000, remaining_trials)
                remaining_trials -= pool_size

                pool = Pool()
                sim_results = [pool.apply_async(self.get_sample_main_event, (
                    groups, matches, bracket)) for _ in range(pool_size)]
                for sim_result in sim_results:
                    ranks = sim_result.get()
                    for rank, teams in ranks.items():
                        for team in teams:
                            final_rank_probs[team][rank] += 1/n_trials
                pbar.update(pool_size)

        if not self.static_ratings:
            self.model = model
        return (group_rank_probs, tiebreak_probs,
                final_rank_probs, record_probs, point_rank_probs)
