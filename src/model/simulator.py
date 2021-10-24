"""This module contains simulators for various tournament formats. Each
Simulator takes a forecasting model as an argument and contains a
simulate() method which simulates one instance and returns the results.

Note that the model is modified in place, so for Monte-Carlo sampling
it is necessary to pass a copy of the model to ensure each simulation
works with a fresh copy. This is not an issue if multiprocessing is
used because the model has to be copied for each process anyway.
"""
import random

from model.tiebreakers import Tiebreaker

class Simulator:
    """Generic simulator class to be subclassed by more specific
    simulator code. Contains utilities for simulating best of n
    matches.

    Parameters
    ----------
    model : TeamModel or Glicko2Model
        Forecasting model to compute win probabilities and team
        ratings.
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
    def __init__(self, model, static_ratings=False):
        self.model = model
        self.static_ratings = static_ratings

    def sim_bo1(self, team1, team2):
        """Simulates a best-of-1 match.

        Parameters
        ----------
        team1 : hashable
            Team 1 identifier (string team name or integer team ID)
        team2 : hashable
            Team 2 identifier (string team name or integer team ID)

        Returns
        -------
        bool
            True if team 1 wins, False if team 2 wins
        """
        win_p_t1 = self.model.get_win_prob(team1, team2)
        team1_win = random.random() < win_p_t1
        if not self.static_ratings:
            self.model.update_ratings(team1, team2, (team1_win, 1 - team1_win))
        return team1_win

    def sim_bo2(self, team1, team2, momentum=0.0):
        """Simulates a best-of-2 match.

        Parameters
        ----------
        team1 : hashable
            Team 1 identifier (string team name or integer team ID)
        team2 : hashable
            Team 2 identifier (string team name or integer team ID)
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
        team1_wins = 0
        team2_wins = 0
        win_p_t1 = self.model.get_win_prob(team1, team2)
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
            self.model.update_ratings(team1, team2, (team1_wins, team2_wins))
        return (team1_wins, team2_wins)

    def sim_bo_n(self, n, team1, team2):
        """Simulates a best-of-n match, where n is odd.

        Parameters
        ----------
        n : int
            Number of games a team must win the majority of to win the
            match.
        team1 : hashable
            Team 1 identifier (string team name or integer team ID)
        team2 : hashable
            Team 2 identifier (string team name or integer team ID)

        Returns
        -------
        tuple of (int, int)
            Number of wins for (team 1, team 2)
        """
        if n % 2 != 1:
            raise ValueError("n must be odd")

        team1_wins = 0
        team2_wins = 0
        win_p_t1 = self.model.get_win_prob(team1, team2)
        while team1_wins < n/2 and team2_wins < n/2:
            team1_win = random.random() < win_p_t1
            team1_wins += int(team1_win)
            team2_wins += 1 - int(team1_win)
        if not self.static_ratings:
            self.model.update_ratings(team1, team2, (team1_wins, team2_wins))
        return (team1_wins, team2_wins)

class TIEliminationBracket(Simulator):
    """16-team double elimination bracket Simulator. Seeds assume a TI
    group stage format

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
        """loads user-defined bracket"""
        self.bracket = bracket

    def _get_winner(self, round_name, n, i):
        """Private helper function which gets the winner of a match
        from the bracket if the match has happens and simulates it
        otherwise.
        """
        match = self.bracket[round_name][i]
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
                        self.model.update_ratings(match[0], match[1], result)
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
