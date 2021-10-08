"""This module contains code for computing TI10 group stage
probabilities using Monte-Carlo sampling. Multiprocessing is used to
speed up computation of results.
"""
from itertools import combinations
from multiprocessing import Pool
import random
import copy
import json

from tqdm import tqdm

from forecaster import PlayerModel, TeamModel
from match_data import MatchDatabase

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
    def from_match_data(cls, roster_file, match_db_file, max_tier,
            k, p=1.5, static_ratings=False, stop_after=None):
        """Constructs and returns a Simulator object by generating a
        player model using match history.

        Parameters
        ----------
        roster_file : str
            Path to JSON file mapping team names to player IDs. The
            file should look something like this:
            {
              "Team A": [1, 2, 3, 4, 5],
              "Team B": [6, 7, 8, 9, 10]
            }
        match_db_file : str, default=None
            Path to sqlite database containing match data. This will be
            used to calculate initial team ratings. If not provided,
            the Elo model will be initialized by adding all players in
            the team rosters using the default rating of 1500.
        max_tier : int, default=2
            If match database is provided: maximum tier of matches to
            grab from match database
        k : int, default=20
            k parameter for player model. See model documentation for
            details
        p : float, default=1.5
            p parameter for player model. See model documentation for
            details
        static_ratings : bool
            (See: constructor documentation above)
        stop_after : int, default=None
            If provided, the model will ignore matches played after
            this date (provided as a unix timestamp)

        Returns
        -------
        Simulator
            Constructed simulator object
        """
        match_db = MatchDatabase(match_db_file)
        player_ids = match_db.get_player_ids()
        p_model = PlayerModel(player_ids, k, p)
        p_model.compute_ratings(match_db.get_matches(max_tier),
            stop_after=stop_after)

        with open(roster_file) as roster_f:
            rosters = json.load(roster_f)
        model = TeamModel.from_player_model(p_model, rosters)

        return cls(model, "team", rosters, static_ratings=static_ratings)

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
        k : int, default=20
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

    def sim_bo2(self, team1, team2):
        """Simulates a best-of-2 match.

        Parameters
        ----------
        team1 : str or list of int
            Team name (TeamModel) or list of player IDs (PlayerModel).
        team2 : str or list of int
            Team name (TeamModel) or list of player IDs (PlayerModel).

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
            output_f.write("}")

class EliminationBracket(Simulator):
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
                [seeds["a"][0], seeds["b"][2 + picks[0]]],
                [seeds["b"][1], seeds["a"][2 + (1 - picks[1])]],
                [seeds["b"][0], seeds["a"][2 + picks[1]]],
                [seeds["a"][1], seeds["b"][2 + (1 - picks[0])]],
            ],
            "UB-R2": [[None, None], [None, None]],
            "UB-F": [None, None],
            "LB-R1": [
                [seeds["a"][4], seeds["b"][6 + picks[2]]],
                [seeds["b"][5], seeds["a"][6 + (1 - picks[3])]],
                [seeds["b"][4], seeds["a"][6 + picks[3]]],
                [seeds["a"][5], seeds["b"][6 + (1 - picks[2])]],
            ],
            "LB-R2": [[None, None], [None, None], [None, None], [None, None]],
            "LB-R3": [[None, None], [None, None]],
            "LB-R4": [[None, None], [None, None]],
            "LB-R5": [None, None],
            "LB-F": [None, None],
            "GF": [None, None]
        }

    def simulate(self):
        """Simulates every game in the bracket and returns the
        resulting placement of each team.

        Returns
        -------
        ranks : dict
            dictionary mapping placement (in the form of a string) to
            teams which received that placement. For consistency all
            dict values are sets even for ranks that only can contain
            one team
        """
        ranks = {"13-16": set(), "9-12": set(), "7-8": set(), "5-6": set(),
                 "4": set(), "3": set(), "2": set(), "1": set()}
        for i, match in enumerate(self.bracket["UB-R1"]):
            winner = int(self.sim_bo_n(3, match[0], match[1])[0] != 2)
            self.bracket["UB-R2"][i//2][i % 2] = match[winner]
            self.bracket["LB-R2"][i][0] = match[1 - winner]
        for i, match in enumerate(self.bracket["LB-R1"]):
            winner = int(self.sim_bo_n(3, match[0], match[1])[0] != 2)
            self.bracket["LB-R2"][i][1] = match[winner]
            ranks["13-16"].add(match[1 - winner])
        for i, match in enumerate(self.bracket["LB-R2"]):
            winner = int(self.sim_bo_n(3, match[0], match[1])[0] != 2)
            self.bracket["LB-R3"][i//2][i % 2] = match[winner]
            ranks["9-12"].add(match[1 - winner])
        for i, match in enumerate(self.bracket["UB-R2"]):
            winner = int(self.sim_bo_n(3, match[0], match[1])[0] != 2)
            self.bracket["UB-F"][i] = match[winner]
            self.bracket["LB-R4"][1 - i][0] = match[1 - winner]
        for i, match in enumerate(self.bracket["LB-R3"]):
            winner = int(self.sim_bo_n(3, match[0], match[1])[0] != 2)
            self.bracket["LB-R4"][i][1] = match[winner]
            ranks["7-8"].add(match[1 - winner])
        for i, match in enumerate(self.bracket["LB-R4"]):
            winner = int(self.sim_bo_n(3, match[0], match[1])[0] != 2)
            self.bracket["LB-R5"][i] = match[winner]
            ranks["5-6"].add(match[1 - winner])
        match = self.bracket["UB-F"]
        winner = int(self.sim_bo_n(3, match[0], match[1])[0] != 2)
        self.bracket["GF"][0] = match[winner]
        self.bracket["LB-F"][0] = match[1 - winner]

        match = self.bracket["LB-R5"]
        winner = int(self.sim_bo_n(3, match[0],match[1])[0] != 2)
        self.bracket["LB-F"][1] = match[winner]
        ranks["4"].add(match[1 - winner])

        match = self.bracket["LB-F"]
        winner = int(self.sim_bo_n(3, match[0], match[1])[0] != 2)
        self.bracket["GF"][1] = match[winner]
        ranks["3"].add(match[1 - winner])

        match = self.bracket["GF"]
        winner = int(self.sim_bo_n(5, match[0], match[1])[0] != 3)
        ranks["2"].add(match[1 - winner])
        ranks["1"].add(match[winner])
        return ranks

class GroupStage(Simulator):
    """Group Stage Simulator. Most of the code in this class is fairly
    general but for now it is specifically written for a TI group stage
    format.

    Parameters
    ----------
    Identical to Simulator. See above for details
    """

    def _order_teams(self, teams, point_map):
        """Given a list of teams and a mapping from team names to
        points, creates an ordered list of teams where tied teams
        occupy the same element of the list as a set.

        Parameters
        ----------
        teams : list of str
            list of team names
        point_map : dict
            Mapping between team names and point values

        Returns
        -------
        list
            Ordered list of teams. If two teams have the same number of
            points, those teams are placed in a set at the same
            position of the list. Example (where B and C are tied):

                ["Team A", {"Team B", "Team C"}, "Team D"]
        """
        team_order = []
        for team in sorted(teams, key=lambda t: -point_map[t]):
            if len(team_order) != 0 and isinstance(team_order[-1], set):
                if point_map[next(iter(team_order[-1]))] == point_map[team]:
                    team_order[-1].add(team)
                else:
                    team_order.append(team)
            elif (len(team_order) != 0
                  and point_map[team_order[-1]] == point_map[team]):
                team_order.append({team_order.pop(), team})
            else:
                team_order.append(team)
        return team_order

    def bo1_tiebreak(self, boundary, teams):
        """Breaks a tie along a given boundary using round-robin bo1s.
        Only ties along the boundary are considered, so in the case of
        a multi-way tie (which is when this tiebreak method is used at
        TI) even if teams aren't completely ordered tiebreaks will stop
        once the teams along the boundary are ordered.

        Parameters
        ----------
        boundary : tuple of (int, int)
            Individual boundary to break a tie at. See documentation
            for boundary_tiebreak for more information
        teams : list of str
            list of team names
        """
        tiebreak_points = {team: 0 for team in teams}
        matches = list(combinations(teams, 2))

        random.shuffle(matches)
        for match in matches:
            team1_win = self.sim_bo1(match[0], match[1])
            tiebreak_points[match[1 - int(team1_win)]] += 1
            tiebreak_points[match[int(team1_win)]] -= 1

            # a 3-way tiebreaker may be cut off early if a team goes
            # 2-0 or 0-2 in the first two matches and such a result
            # resolves the tie without needing the third match
            # (see TI8 for an example)
            # in theory something similar could happen with a 4+ way
            # tie but matches might be played simultaneously and there
            # isn't any precedence to determine whether the matches
            # would end early anyway so I explicitly only stop early
            # for 3-way ties
            if len(teams) == 3:
                if (tiebreak_points[match[1-int(team1_win)]] == len(teams) - 1
                      and boundary[0] == 0):
                    break
                if (tiebreak_points[match[int(team1_win)]] == 1 - len(teams)
                      and boundary[1] == len(teams) - 1):
                    break

        team_order = self._order_teams(teams, tiebreak_points)

        rank_start = 0
        for i, team_sets in enumerate(team_order):
            if not isinstance(team_sets, set):
                rank_start += 1
                continue
            rank_end = rank_start + len(team_sets) - 1
            if rank_start <= boundary[0] and rank_end >= boundary[1]:
                break_pos = boundary[0] - rank_start
                relative_boundary = (break_pos, break_pos + 1)
                # if there's still a tie along the boundary,
                # play more bo1s to resolve it
                team_order[i:i+1] = self.bo1_tiebreak(
                    relative_boundary, team_sets)
            rank_start += len(team_sets)

        return team_order

    def boundary_tiebreak(self, tiebreak_boundaries, team_order, point_map):
        """Breaks ties along a boundary by playing additional matches.
        Once the boundary tie has been broken, no additional matches
        are played. Following official TI rules, a bo3 is used for
        2-way ties and bo1s are used for 3+ way ties.

        Ties which aren't along a boundary which requires tiebreak
        matches are not considered (these ties use head-to-head
        results, which are covered by a different function)

        Parameters
        ----------
        tiebreak_boundaries : List of tuple of int
            List of positions (0-indexed) which require additional
            matches in the case of a tie. For example, in the case of
            TI (4th/5th and 8th/9th require tiebreakers) this would be:

                [(3,4), (7,8)]

            The second number is technically redundant but I find it
            more intuitive/easier to remember
        team_order : list
            Team ordering as returned by _order_teams
        point_map : dict
            Mapping from team name to current points.

        Returns
        -------
        team_order : list
            Updated team ordering with boundary ties resolved
        tiebreak_sizes : list of int
            List containing the size of each broken tie
        """
        num_groups = len(team_order) # number of tied groups

        rank_end = len(point_map) - 1
        tiebreak_sizes = {boundary[0]: [] for boundary in tiebreak_boundaries}
        for i, teams in enumerate(reversed(team_order)):
            # no tie:
            if not isinstance(teams, set):
                rank_end -= 1
                continue

            # determine if tie is along a boundary for which
            # additional matches are played
            boundary_tie = False
            rank_start = rank_end - (len(teams) - 1)
            for boundary in tiebreak_boundaries:
                if rank_start <= boundary[0] and rank_end >= boundary[1]:
                    boundary_tie = True
                    break
            if not boundary_tie:
                rank_end -= len(teams)
                continue

            # there is a tie, and it is along a boundary
            if len(teams) == 2:
                # 2-way tiebreak, play a single bo3
                teams = list(teams)
                result = self.sim_bo_n(3, teams[0], teams[1])
                if result[0] == 2: # team 1 victory
                    team_order[num_groups - i-1:num_groups - i] = [
                        teams[0], teams[1]]
                else:
                    team_order[num_groups - i-1:num_groups - i] = [
                        teams[1], teams[0]]
                rank_end -= 2
                tiebreak_sizes[boundary[0]].append(2)
            else:
                # multi-way tiebreak, play bo1s until tiebreak resolved
                break_pos = boundary[0] - rank_start
                relative_boundary = (break_pos, break_pos + 1)
                reordered = self.bo1_tiebreak(relative_boundary, teams)
                team_order[num_groups - i-1:num_groups - i] = reordered

                rank_end -= len(teams)
                tiebreak_sizes[boundary[0]].append(len(teams))

        return team_order, tiebreak_sizes

    def h2h_tiebreak(self, h2h_results, team_order, point_map):
        """Breaks any remaining ties in an ordering using head-to-head
        results. Following TI rules, if head-to-head results do not
        resolve the tie results against lower seeded teams are
        considered. If these do not resolve the tie, seeding is
        assigned at random (TI rules specify a coin toss, but it is
        technically possible to have a 3-way unresolved tie).

        Parameters
        ----------
        h2h_results : dict
            dict containing head-to-head results. Example:
            {
                "Team A": {"Team B": 2, "Team C": 1},
                "Team B": {"Team A": 0, "Team C": 2},
                "Team C": {"Team A": 1, "Team B": 0},
            }
            Note that the table should be complete, even though this
            results in redundant information. The memory cost is not
            significant and it makes the code a bit simpler.
        team_order : list
            Team ordering as returned by _order_teams
        point_map : dict
            Mapping from team name to current points.

        Returns
        -------
        team_order : list
            Updated team ordering with boundary ties resolved
        """
        # as long as the team order contains a set (unbroken tie),
        # it will have fewer items than the total number of teams
        while len(team_order) != len(point_map):
            # ties must be broken bottom-up because if head-to-head results
            # don't break ties results against lower-seeded teams do, so
            # lower-seeded ties must be broken first
            num_groups = len(team_order)
            for i, teams in enumerate(reversed(team_order)):
                if not isinstance(teams, set):
                    continue
                tiebreak_points = {team: 0 for team in teams}
                # first, check head-to-head
                for (team1, team2) in combinations(teams, 2):
                    tiebreak_points[team1] += h2h_results[team1][team2]
                    tiebreak_points[team2] += h2h_results[team2][team1]

                tie_order = self._order_teams(teams, tiebreak_points)
                if len(tie_order) != 1:
                    team_order[num_groups - i - 1:num_groups - i] = tie_order
                    break # any time a tie is broken, the process restarts

                # head-to-head is even, so check results against lower seeds
                next_seed = len(team_order) - i
                tie_broken = False
                while next_seed < len(team_order):
                    team2 = team_order[next_seed]
                    for team in teams:
                        tiebreak_points[team] += h2h_results[team][team2]
                    tie_order = self._order_teams(teams, tiebreak_points)
                    if len(tie_order) != 1:
                        tie_broken = True
                        break
                    next_seed += 1

                if tie_broken:
                    team_order[num_groups - i - 1:num_groups - i] = tie_order
                    break # any time a tie is broken, the process restarts

                # tie can't be broken, assign at random
                tied_teams = list(teams)
                random.shuffle(tied_teams)
                team_order[num_groups - i - 1:num_groups - i] = tied_teams

        return team_order

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
        points : list of list(str, int)
            Sorted list of teams and points. Example:
                [["Team A", 10], ["Team B", 9], ["Team C", 5]]
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
                    result = self.sim_bo2(match[0], match[1])
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
        team_order = self._order_teams(point_map.keys(), point_map)

        team_order, tiebreak_sizes = self.boundary_tiebreak([(3,4), (7,8)],
            team_order, point_map)
        team_order = self.h2h_tiebreak(h2h_results,
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
                Ordered (team, points) tuple for each group
            dict
                Size of tiebreak required along each boundary
            dict
                Mapping from ranks to the teams placed at that rank
        """
        if not self.static_ratings:
            # if ratings are updated during a simulation, the code
            # needs to work with a copy to make sure each simulation
            # is fresh
            working_model = copy.deepcopy(self.model)
        else:
            working_model = self.model
        gs_sim = GroupStage(working_model, "team",
                            self.rosters, self.static_ratings)

        points_a, tiebreak_sizes_a = gs_sim.simulate(groups["a"], matches["a"])
        points_b, tiebreak_sizes_b = gs_sim.simulate(groups["b"], matches["b"])

        bracket_sim = EliminationBracket(gs_sim.model, "team",
                                         self.rosters, self.static_ratings)
        bracket_sim.seed({"a": [p[0] for p in points_a],
                          "b": [p[0] for p in points_b]})
        ranks = bracket_sim.simulate()

        return ({"a": points_a, "b": points_b},
                {"a": tiebreak_sizes_a, "b": tiebreak_sizes_b},
                ranks)

    def sim_group_stage(self, groups, matches, n_trials):
        """Wrapper for running sample_group_stage many many times.
        uses multiprocessing to speed up results, which it combines
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
                        for boundary, sizes in tiebreak_sizes[group].items():
                            for size in sizes:
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
