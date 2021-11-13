"""This module contains the Monte-Carlo samplers used for computing
probability estimates. Multiprocessing is used to speed up computation
of results.
"""

from multiprocessing import Pool
import json
import copy
from itertools import combinations
import random

from tqdm import tqdm

from model.simulator import (TIEliminationBracket, TIGroupStage,
                             DPCLeague, DPCMajor)
from model.forecaster import TeamModel
from model.forecaster_glicko import Glicko2Model

class Sampler:
    """Generic sampler class to be subclassed by samplers for specific
    events. Contains utilities for getting bo2 probabilities and
    initializing the sampler with a model.

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
    """
    def __init__(self, model, static_ratings=False):
        self.model = model
        self.static_ratings = static_ratings

    def get_bo2_probs(self, team1, team2, draw_adjustment=0.0):
        """Computes win/draw/loss probabilities for a bo2 match.

        Parameters
        ----------
        team1 : hashable
            Team 1 identifier (string team name or integer team ID)
        team2 : hashable
            Team 2 identifier (string team name or integer team ID)
        draw_adjustment : float, default=0.0
            Using game probabilities alone results in consistent over-
            estimation of draw probabilities in bo2s. This counteracts
            that error by reducing draw change by the given amount.

        Returns
        -------
        tuple of float
            Probability of 2-0, 1-1, and 0-2 results (in that order).
            Should always sum to 1
        """
        win_p_t1 = self.model.get_win_prob(team1, team2)
        if not draw_adjustment:
            p2_0 = pow(win_p_t1, 2)
            p0_2 = pow(1 - win_p_t1, 2)
        else:
            # winner probability adjustment shouldn't exceed 95%
            p2_0 = win_p_t1*min(max(0.95, win_p_t1), win_p_t1+draw_adjustment)
            p0_2 = (1 - win_p_t1)*min(max(0.95, 1 - win_p_t1),
                                      (1 - win_p_t1) + draw_adjustment)
        return (p2_0, (1 - (p2_0 + p0_2)), p0_2)

    @classmethod
    def from_ratings_file(cls, ratings_file, k, static_ratings=False):
        """Constructs and returns a Sampler object from a file
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
        Sampler
            Constructed sampler object
        """
        with open(ratings_file) as rating_f:
            ratings = json.load(rating_f)
        model = TeamModel(ratings, k)
        return cls(model, static_ratings=static_ratings)

    @classmethod
    def from_ratings_file_glicko2(cls, ratings_file, tau,static_ratings=False):
        """Constructs and returns a Sampler object from a file
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
        Sampler
            Constructed sampler object
        """
        with open(ratings_file) as rating_f:
            ratings = json.load(rating_f)
        model = Glicko2Model(tau)
        for team, rating_tuple in ratings.items():
            model.ratings[team] = ((rating_tuple[0] - 1500)/173.7178,
                rating_tuple[1]/173.7178, rating_tuple[2])
        return cls(model, static_ratings=static_ratings)

class TISampler(Sampler):
    """TI Sampler. For each Monte Carlo sample a GroupStage object and
    an EliminationBracket object are created using a working copy of
    the prediction model. Multiprocessing is then used to collect many
    samples reasonably quickly.

    Parameters
    ----------
    Identical to Sampler. See above for details
    """

    @staticmethod
    def get_sample(model, groups, matches, static_ratings):
        """Gets a single sample of group stage placements and final
        ranks using the GroupStage and EliminationBracket simulators.

        Parameters
        ----------
        model: TeamModel or Glicko2Model
            Copy of self.model. The sampler is a static function for
            multiprocessing efficiency reasons, so this needs to be
            passed explicitly.
        groups : dict
            (See: sample_group_stage documentation below)
        matches : dict
            (See: sample_group_stage documentation below)
        static_ratings : bool
            Copy of self.static_ratings.

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
        gs_sim = TIGroupStage(model, static_ratings)

        points_a, tiebreak_sizes_a = gs_sim.simulate(groups["a"], matches["a"],
            matches.get("tiebreak", {}).get("a", {}))
        points_b, tiebreak_sizes_b = gs_sim.simulate(groups["b"], matches["b"],
            matches.get("tiebreak", {}).get("b", {}))

        bracket_sim = TIEliminationBracket(gs_sim.model, static_ratings)
        bracket_sim.seed({"a": [p[0] for p in points_a],
                          "b": [p[0] for p in points_b]})
        ranks = bracket_sim.simulate()

        return ({"a": points_a, "b": points_b},
                {"a": tiebreak_sizes_a, "b": tiebreak_sizes_b},
                ranks)

    @staticmethod
    def get_sample_main_event(model, bracket, static_ratings):
        """Gets a single sample of final ranks from a single
        EliminationBracket simulation.

        Parameters
        ----------
        model: TeamModel or Glicko2Model
            Copy of self.model. The sampler is a static function for
            multiprocessing efficiency reasons, so this needs to be
            passed explicitly.
        matches : dict
            (See: sample_group_stage documentation below)
        bracket : dict
            Bracket with team seeds and match results.
        static_ratings : bool
            Copy of self.static_ratings.

        Returns
        -------
            dict
                Mapping from ranks to the teams placed at that rank
        """
        bracket_sim = TIEliminationBracket(model, static_ratings)
        bracket_sim.load_bracket(bracket)
        ranks = bracket_sim.simulate()

        return ranks

    def sample_group_stage(self, groups, matches, n_trials):
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
                "4": 0, "3": 0, "2": 0, "1": 0}
            for team in self.model.ratings.keys()
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
                        self.model, groups, matches, self.static_ratings))
                    for _ in range(pool_size)]
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
                            if size != 0:
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
        return {
            "group_rank": group_rank_probs,
            "tiebreak": tiebreak_probs,
            "final_rank": final_rank_probs,
            "record": record_probs,
            "point_rank": point_rank_probs
        }

    def sample_main_event(self, groups, matches, bracket, n_trials):
        """Wrapper for running get_sample_main_event many times.
        Arguments and return values are identical to sample_group_stage
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
                "4": 0, "3": 0, "2": 0, "1": 0}
            for team in self.model.ratings.keys()
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
        points, tiebreak_sizes, ranks = self.get_sample(
            copy.deepcopy(self.model), groups, matches, self.static_ratings)
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
        # happen seperately. samplers aren't  supposed to update the
        # model so a copy has to be saved so it can be restored after
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
                        self.model, bracket, self.static_ratings))
                    for _ in range(pool_size)]
                for sim_result in sim_results:
                    ranks = sim_result.get()
                    for rank, teams in ranks.items():
                        for team in teams:
                            final_rank_probs[team][rank] += 1/n_trials
                pbar.update(pool_size)

        if not self.static_ratings:
            self.model = model
        return {
            "group_rank": group_rank_probs,
            "tiebreak": tiebreak_probs,
            "final_rank": final_rank_probs,
            "record": record_probs,
            "point_rank": point_rank_probs
        }

class DPCLeagueSampler(Sampler):
    """DPC League sampler for the 2020 - 2021 season format. Can be
    used to obtain results for any of the 6 regions through
    configuration of the wildcard_slots parameter.

    Parameters
    ----------
    Identical to Sampler. See above for details
    """

    @staticmethod
    def get_sample(model, matches, teams, wildcard_slots, static_ratings):
        """Gets a single sample of DPC league placement.

        Parameters
        ----------
        model: TeamModel or Glicko2Model
            Copy of self.model. The sampler is a static function for
            multiprocessing efficiency reasons, so this needs to be
            passed explicitly.
        matches : dict
            (See: sample_league documentation below)
        teams : dict
            (See: sample_league documentation below)
        wildcard_slots : int, {0, 1, 2}
            Number of wildcard slots provided to the league.
        static_ratings : bool
            Copy of self.static_ratings.

        Returns
        -------
            dict
                Ordered (team, points) tuple for each group.
            dict
                Size of tiebreak required along each boundary. 0 if no
                tiebreaker matches were necessary.
        """
        if wildcard_slots > 0:
            upper_div_sim = DPCLeague(model, [(0,1), (1,2),
                (1+wildcard_slots, 2+wildcard_slots), (5,6)], static_ratings)
        else:
            upper_div_sim = DPCLeague(model, [(0,1), (1,2), (5,6)],
                                      static_ratings)
        points_upper, tiebreak_sizes_upper = upper_div_sim.simulate(
            teams["upper"], matches["upper"],
            matches.get("tiebreak", {}).get("upper", {}))

        lower_div_sim = DPCLeague(model, [(1,2), (5,6)], static_ratings)
        points_lower, tiebreak_sizes_lower = lower_div_sim.simulate(
            teams["lower"], matches["lower"],
            matches.get("tiebreak", {}).get("lower", {}))

        return ({"upper": points_upper, "lower": points_lower},
                {"upper": tiebreak_sizes_upper, "lower": tiebreak_sizes_lower})

    def sample_league(self, teams, matches, wildcard_slots, n_trials):
        """Gets many samples of league placement and returns aggregate
        probabilities of league ranks and tiebreaker probabilities

        Parameters
        ----------
        teams : dict
            List of teams in each division:
            {
                "upper": ["Team A", "Team B"],
                "lower": ["Team C", "Team D"]
            }
        matches : dict
            List of matches for each division. Each match is a 3-
            element list containing team 1, team 2, and the match
            result as a pair. The pair is empty if the match hasn't
            happened yet. Matches must be provided as a lists of lists,
            where each sub-list contains the matches for a single day.

            Tiebreaker matches should be provide separately as a
            dictionary mapping each boundary tiebreakers were played
            along to the results of those tiebreakers.

            Example:
            {
                "upper": [[
                    ["Team A", "Team B", [0, 2]],
                    ["Team B", "Team C", [1, 2]]
                ]],
                "lower": [[
                    ["Team D", "Team E", [2, 1]],
                    ["Team E", "Team D", []]
                ]],
                "tiebreak" :{
                    "upper": {
                        "5": [["Team B", "Team C", [2, 0]]]
                    }
                    "lower": {}
                }
            }
            In this case the results are A 0-2 B, B 1-2 C, D 2-1 E, and
            E vs D has not yet been played.
            An additional match between B and C was played to break the
            tie along spots 5-6 (0-indexed, so this would be between
            ranks 6 and 7). Note that this is in a nested list because
            there may be multiple tiebreak matches (e.g., for a 3-way
            tie).
        wildcard_slots : int, {0, 1, 2}
            Number of wildcard slots provided to the league.
        n_trials : int
            Number of simulations to run.
        """
        if wildcard_slots > 0:
            upper_tiebreak_boundaries = [
                (0,1), (1,2), (1+wildcard_slots, 2+wildcard_slots), (5,6)]
        else:
            upper_tiebreak_boundaries = [(0,1), (1,2), (5,6)]
        group_rank_probs = {
            "upper": {team: [0 for _ in range(len(teams["upper"]))]
                  for team in teams["upper"]},
            "lower": {team: [0 for _ in range(len(teams["lower"]))]
                  for team in teams["lower"]}}
        tiebreak_probs = {
            "upper": {boundary: [0 for _ in range(len(teams["upper"]) - 1)]
                  for boundary in [b[0] for b in upper_tiebreak_boundaries]},
            "lower": {boundary: [0 for _ in range(len(teams["lower"]) - 1)]
                  for boundary in [1,5]}}
        point_rank_probs = {
            group: {
                team: {
                    points: {
                        rank: 0 for rank in range(len(teams[group]))
                    } for points in range(len(teams[group])*2 - 1)
                } for team in teams[group]
            } for group in ["upper", "lower"] }
        record_probs = {group: {team: [0 for _ in range(len(teams[group])*2-1)]
                    for team in teams[group]} for group in ["upper", "lower"]}

        # all results are stored in memory until the pool completes,
        # so pool size is limited to 1,000 to reduce memory usage
        remaining_trials = n_trials
        with tqdm(total=n_trials) as pbar:
            while remaining_trials > 0:
                pool_size = min(1000, remaining_trials)
                remaining_trials -= pool_size

                pool = Pool()
                sim_results = [pool.apply_async(self.get_sample, (
                        self.model, matches, teams, wildcard_slots,
                        self.static_ratings))
                    for _ in range(pool_size)]
                for sim_result in sim_results:
                    points, tiebreak_sizes = sim_result.get()
                    for division in ["upper", "lower"]:
                        for i, (team, record) in enumerate(points[division]):
                            group_rank_probs[division][team][i] += 1/n_trials
                            point_rank_probs[division][team][record][i] += 1
                            record_probs[division][team][record] += 1/n_trials
                        for bound,size in tiebreak_sizes[division].items():
                            if size != 0:
                                tiebreak_probs[division][bound][
                                    size - 2] += 1/n_trials
                pbar.update(pool_size)

        for group in ["upper", "lower"]:
            for team, record_map in point_rank_probs[group].items():
                for record, point_counts in record_map.items():
                    points_sum = sum(point_counts.values())
                    for points, amount in point_counts.items():
                        if amount > 0:
                            point_counts[points] = amount / points_sum

        return {
            "group_rank": group_rank_probs,
            "tiebreak": tiebreak_probs,
            "record": record_probs,
            "point_rank": point_rank_probs
        }

class DPCMajorSampler(Sampler):
    """DPC major sampler for the 2020 - 2021 season format. Includes
    all three stages of the major (wildcard, group stage, playoffs)

    Parameters
    ----------
    Identical to Sampler. See above for details
    """

    @staticmethod
    def get_sample(model, matches, teams, static_ratings):
        """Gets a single sample of a DPC major (wildcard, group stage,
        and playoffs).

        Parameters
        ----------
        model: TeamModel or Glicko2Model
            Copy of self.model. The sampler is a static function for
            multiprocessing efficiency reasons, so this needs to be
            passed explicitly.
        matches : dict
            (See: sample_league documentation below)
        teams : dict
            (See: sample_league documentation below)
        static_ratings : bool
            Copy of self.static_ratings.

        Returns
        -------
            dict
                Mapping from rank to teams placed at that rank. For
                consistency, all values are sets even though most ranks
                only contain 1 team.
        """
        final_ranks = {"18": None, "17": None, "16": None, "15": None,
            "14": None, "13": None, "9-12": None, "7-8": None,"5-6": None,
            "4": None, "3": None, "2": None, "1": None}

        sim = DPCMajor(model, static_ratings)
        points, _ = sim.sim_wildcard(teams["wildcard"], matches["wildcard"],
            matches.get("tiebreak", {}).get("wildcard", {}))
        wildcard_map = {}
        for i, (team, score) in enumerate(points):
            if i < 2:
                wildcard_map[f"TBD/wildcard{i+1}"] = team
                teams["group stage"].append(team)
            else:
                final_ranks[f"{15 + (i - 2)}"] = {team}

        for match_list in matches["group stage"]:
            for match in match_list:
                for i in range(2):
                    if match[i] in wildcard_map:
                        match[i] = wildcard_map[match[i]]

        points, _ = sim.sim_group_stage(teams["group stage"],
            matches["group stage"],
            matches.get("tiebreak", {}).get("group stage", {}))
        groups_map = {}
        for i, (team, score) in enumerate(points):
            if i < 6:
                groups_map[f"TBD/gs{i+1}"] = team
            else:
                final_ranks[f"{13 + (i - 6)}"] = {team}

        for match_list in matches["playoffs"].values():
            for match in match_list:
                for i in range(2):
                    if match[i] in groups_map:
                        match[i] = groups_map[match[i]]

        playoff_ranks = sim.sim_playoffs(matches["playoffs"])
        for rank, team_set in playoff_ranks.items():
            final_ranks[rank] = team_set

        return final_ranks

    def sample_major(self, teams, matches, n_samples):
        """Gets many samples of major placement and returns aggregate
        probabilities of final ranks.

        Parameters
        ----------
        teams : dict
            List of teams at each stage of the major
            {
                "wildcard": ["Team A", "Team B"],
                "group stage": ["Team C", "Team D"],
                "playoffs": ["Team E", "Team F"]
            }
        matches : dict
            List of all matches to be played at the major. The format
            is quite long so instead of documenting it here I refer to
            the file at src/dpc/spring/major/matches.json as an example.

            Note that the dicationary contains 4 keys, the last of
            which is optional (wildcard, group stage, playoffs,
            tiebreak). Match format depends on the stage of the
            competition.
        n_samples : int
            Number of simulations to run.
        """
        all_teams = teams["wildcard"]+teams["group stage"]+teams["playoffs"]
        final_rank_probs = {team: {
                "18": 0, "17": 0, "16": 0, "15": 0, "14": 0,
                "13": 0, "9-12": 0, "7-8": 0,"5-6": 0, "4": 0,
                "3": 0, "2": 0, "1": 0}
            for team in all_teams}

        remaining_trials = n_samples
        with tqdm(total=n_samples) as pbar:
            while remaining_trials > 0:
                pool_size = min(1000, remaining_trials)
                remaining_trials -= pool_size

                with Pool() as pool:
                    sim_results = [pool.apply_async(self.get_sample, (
                            self.model, matches, teams, self.static_ratings))
                        for _ in range(pool_size)]
                    for sim_result in sim_results:
                        final_ranks = sim_result.get()
                        for rank, team_set in final_ranks.items():
                            for team in team_set:
                                final_rank_probs[team][rank] += 1/n_samples

                pbar.update(pool_size)

        return {
            "final_rank": final_rank_probs
        }

class DPCSeasonSampler(Sampler):
    """DPC season sampler. Computes probabilities over an entire DPC
    season consisting of 3 tours. Match schedules are generated
    randomly so unlike the other samplers it is not currently possible
    to load mid-season results.

    Parameters
    ----------
    Identical to Sampler. See above for details
    """

    @staticmethod
    def _get_league_schedule(teams):
        """Randomly generates a DPC league match schedule with a 6-week
        league duration (league duration doesn't really matter, but the
        league simulator assumes matches are split into per-week lists)
        """
        upper_matches = [[match[0], match[1], []]
            for match in combinations(teams["upper"], 2)]
        lower_matches = [[match[0], match[1], []]
            for match in combinations(teams["lower"], 2)]
        random.shuffle(upper_matches)
        random.shuffle(lower_matches)
        matches = {
            "upper": [upper_matches[i*5:(i+1)*5] for i in range(6)],
            "lower": [lower_matches[i*5:(i+1)*5] for i in range(6)],
        }
        return matches

    @staticmethod
    def _get_major_schedule(teams):
        """Randomly generates a DPC major schedule"""
        wildcard_matches = [[match[0], match[1], []]
            for match in combinations(teams["wildcard"], 2)]
        gs_matches = [[match[0], match[1], []]
            for match in combinations(teams["group stage"]
                                  + ["TBD/wildcard1", "TBD/wildcard2"], 2)]
        random.shuffle(wildcard_matches)
        random.shuffle(gs_matches)
        # playoff seeding will probably be based on DPC points but
        # nothing has been confirmed so seeds are random instead
        random.shuffle(teams["playoffs"])
        bracket = {
            "UB-R1": [
              [teams["playoffs"][0], "TBD/gs2", []],
              [teams["playoffs"][1], "TBD/gs1", []],
              [teams["playoffs"][2], teams["playoffs"][3],[]],
              [teams["playoffs"][4], teams["playoffs"][5],[]]],
            "UB-R2": [["", "", []], ["", "", []]], "UB-F": [["", "", []]],
            "LB-R1": [["", "TBD/gs3", []], ["", "TBD/gs6", []],
                      ["", "TBD/gs5", []], ["", "TBD/gs4", []]],
            "LB-R2": [["", "", []], ["", "", []]],
            "LB-R3": [["", "", []], ["", "", []]],
            "LB-R4": [["", "", []]], "LB-F": [["", "", []]],
            "GF": [["", "", []]]}
        matches = {
            "wildcard": [wildcard_matches[:8], wildcard_matches[8:]],
            "group stage": [gs_matches[:8], gs_matches[8:16],
                            gs_matches[16:24], gs_matches[24:]],
            "playoffs": bracket
        }
        return matches

    @staticmethod
    def _get_point_allocation(season):
        """Returns the point allocation per rank for the provided
        season.
        """
        if season == "21-22":
            league_allocation = [
                (300, 400, 500), (180, 240, 300), (120, 160, 200),
                (60, 80, 100), (30, 40, 50)
            ]
            major_allocation = [
                (400, 500, 600), (350, 450, 550), (300, 400, 500),
                (250, 350, 450), (200, 300, 400), (100, 200, 300)
            ]
        elif season == "20-21":
            league_allocation = [
                (500, 500, 500), (300, 300, 300), (200, 200, 200),
                (100, 100, 100), (50, 50, 50)
            ]
            major_allocation = [
                (500, 500, 500), (450, 450, 450), (400, 400, 400),
                (350, 350, 350), (300, 300, 300), (200, 200, 200)
            ]
        else:
            raise ValueError("Invalid season")
        return league_allocation, major_allocation

    @staticmethod
    def _promote_relegate(teams, points):
        """Promotes top two teams from the lower division and relegates
        bottom two teams from the upper division.

        There's no way to know what the open qualifier teams will be,
        so it is assumed that the two teams relegated from lower
        division will qualify back through the following open
        qualifier.

        This assumption has no effect on point probabilities because
        with 3 tours it's impossible to earn DPC points, get relegated
        from the lower division, then earn DPC points again (this would
        require at least 5 tours)
        """
        # upper division relegation
        teams["upper"].remove(points["upper"][6][0])
        teams["upper"].remove(points["upper"][7][0])
        teams["lower"].append(points["upper"][6][0])
        teams["lower"].append(points["upper"][7][0])
        # lower division promotion
        teams["lower"].remove(points["lower"][0][0])
        teams["lower"].remove(points["lower"][1][0])
        teams["upper"].append(points["lower"][0][0])
        teams["upper"].append(points["lower"][1][0])

    @staticmethod
    def _dpc_tiebreak(dpc_points, major_results, league_results):
        """Breaks ties for 12th place (TI qualification) according to
        the rules used for the 2020-2021 season. Specifically:
        - major placement, from most recent to least recent
        - league placement, from most recent to least recent
        """
        points_12th = dpc_points[11][1]
        tie_end = 11
        while dpc_points[tie_end + 1][1] == points_12th:
            tie_end += 1
        if tie_end == 11:
            # no tie between 12th/13th, no additional work needed
            return

        tie_start = 11
        while tie_start > 0 and dpc_points[tie_start - 1][1] == points_12th:
            tie_start -= 1
        tied_teams = dpc_points[tie_start:tie_end + 1]

        # get ranks in majors followed by ranks in leagues, ordered by
        # tour from last to first
        season_ranks = {}
        for team, points in tied_teams:
            season_ranks[(team, points)] = []
            for season in ["summer", "spring", "winter"]:
                season_ranks[(team, points)].append(
                    int(major_results[season].get(team, "20").split("-")[0]))
            for season in ["summer", "spring", "winter"]:
                season_ranks[(team, points)].append(
                    league_results[season].get(team, 9))

        # python's sort function automatically handles lists where
        # element order determines sort priority
        sorted_teams = []
        for ((team, points), ranks) in sorted(season_ranks.items(),
                                              key=lambda x: x[1]):
            sorted_teams.append((team, points))
        dpc_points[tie_start:tie_end + 1] = sorted_teams

    @staticmethod
    def get_sample(model, teams, season, static_ratings):
        """Simulates a single DPC season consisting of 3 tours (6
        regional leagues followed by a major).

        Parameters
        ----------
        model: TeamModel or Glicko2Model
            Copy of self.model. The sampler is a static function for
            multiprocessing efficiency reasons, so this needs to be
            passed explicitly.
        teams : dict
            (See: sample_season documentation below)
        season : {"20-21" "21-22"}
            Determines which point allocation scheme to use.
        static_ratings : bool
            Copy of self.static_ratings.

        Returns
        -------
        list of (str, int)
            Ordered list of tuples containing team name and DPC points
        dict
            Mapping from team name to major placement for each season.
        dict
            Mapping from team name to league placement for each season.
            Only returns results for upper division teams.
        """
        dpc_points = {}
        for region in ["na", "sa", "weu", "eeu", "cn", "sea"]:
            for division in ["upper", "lower"]:
                for team in teams[region][division]:
                    dpc_points[team] = 0
        major_results = {"winter": {}, "spring": {}, "summer": {}}
        league_results = {"winter": {}, "spring": {}, "summer": {}}
        wildcard_slots = {"sea":1, "eeu":1, "cn":2, "weu":2, "na":0, "sa":0}

        league_allocation, major_allocation = (
            DPCSeasonSampler._get_point_allocation(season))

        for tour_idx, tour in enumerate(["winter", "spring", "summer"]):
            # regional league simulation:
            major_teams = {"playoffs": [], "group stage": [], "wildcard": []}
            for region in ["na", "sa", "weu", "eeu", "cn", "sea"]:
                matches = DPCSeasonSampler._get_league_schedule(teams[region])
                points, _ = DPCLeagueSampler.get_sample(model, matches,
                    teams[region], wildcard_slots[region], static_ratings)

                # major seeding
                major_teams["playoffs"].append(points["upper"][0][0])
                major_teams["group stage"].append(points["upper"][1][0])
                if region in ["weu", "eeu", "sea", "cn"]:
                    major_teams["wildcard"].append(points["upper"][2][0])
                if region in ["weu", "cn"]:
                    major_teams["wildcard"].append(points["upper"][3][0])

                DPCSeasonSampler._promote_relegate(teams[region], points)

                # allocate league points
                for i, (team, _) in enumerate(points["upper"]):
                    if i < 5:
                        dpc_points[team] += league_allocation[i][tour_idx]
                    league_results[tour][team] = i

            # major simulation:
            matches = DPCSeasonSampler._get_major_schedule(major_teams)
            major_ranks = DPCMajorSampler.get_sample(model,matches,major_teams,
                                                     static_ratings)
            for rank, team_set in major_ranks.items():
                for team in team_set:
                    major_results[tour][team] = rank

            # allocate major points
            for i, rank in enumerate(["1", "2", "3", "4", "5-6", "7-8"]):
                for team in major_ranks[rank]:
                    dpc_points[team] += major_allocation[i][tour_idx]

        sorted_points = sorted(dpc_points.items(), key=lambda x: -x[1])

        DPCSeasonSampler._dpc_tiebreak(sorted_points, major_results,
                                       league_results)
        return sorted_points, major_results, league_results

    def _welford_update(self, mean, mean_diff_sum, val, match_count):
        """Computes the new running mean and running mean squared
        difference according to Welford's algorithm for computing mean
        and variance in one pass.
        """
        new_mean = mean + (val - mean)/match_count
        new_mean_diff_sum = mean_diff_sum + (val - mean)*(val - new_mean)
        return [new_mean, new_mean_diff_sum]

    def _get_major_contribution(self, teams, major_results, season):
        """Determines how many points each team earned from majors.
        This could easily be calculated in get_sample but it's not
        needed for most cases and increases the multiprocessing load.
        """
        point_value = {}
        _, major_allocation = self._get_point_allocation(season)
        for tour_idx, tour in enumerate(["winter", "spring", "summer"]):
            point_value[tour] = {}
            for i, rank in enumerate(["1", "2", "3", "4", "5-6", "7-8"]):
                point_value[tour][rank] = major_allocation[i][tour_idx]

        contributions = []
        for team in teams:
            major_contribution = 0
            for tour in ["winter", "spring", "summer"]:
                rank = major_results[tour].get(team, "")
                major_contribution += point_value[tour].get(rank, 0)
            contributions.append(major_contribution)
        return contributions

    def sample_season(self, teams, season, n_samples):
        """Gets many samples of a full DPC season given a list of teams
        for each region.

        Parameters
        ----------
        teams : dict
            List of teams in each region/division. Example:
            {
                "na": {
                    "upper": ["Team A", "Team B"],
                    "lower": ["Team C", "Team D"]
                },
                "sa": {
                    "upper": ["Team C", "Team D"],
                    "lower": ["Team E", "Team F"]
                }
            }
        season : {"20-21" "21-22"}
            Determines which point allocation scheme to use.
        n_samples : int
            Number of simulations to run.
        """
        all_teams = []
        for region in ["na", "sa", "weu", "eeu", "cn", "sea"]:
            for division in ["upper", "lower"]:
                for team in teams[region][division]:
                    all_teams.append(team)

        # mean/variance of team ranks
        final_rank_probs = {team: [0, 0] for team in all_teams}
        # rank/point joint probability distribution
        rank_point_dist = [{} for rank in range(96)]
        # probability of obtaining x points from a major
        major_contrib_ests = {threshold: 0 for threshold in range(0,2000,100)}

        remaining_trials = n_samples
        sim_count = 1
        with tqdm(total=n_samples) as pbar:
            while remaining_trials > 0:
                pool_size = min(1000, remaining_trials)
                remaining_trials -= pool_size

                with Pool() as pool:
                    sim_results = [pool.apply_async(self.get_sample, (
                            self.model, teams, season, self.static_ratings))
                        for _ in range(pool_size)]
                    for sim_result in sim_results:
                        dpc_points, major_results, _ = sim_result.get()
                        ti_teams = []
                        for i, (team, points) in enumerate(dpc_points):
                            final_rank_probs[team] = self._welford_update(
                                *final_rank_probs[team], points, sim_count)
                            rank_point_dist[i][points]=rank_point_dist[i].get(
                                points, 0) + 1/n_samples/96
                            if i < 12:
                                ti_teams.append(team)

                        major_contribution = self._get_major_contribution(
                            ti_teams, major_results, season)
                        for point_thresh in major_contrib_ests.keys():
                            major_contrib_ests[point_thresh] += sum(
                                [points <= point_thresh
                                 for points in major_contribution])/n_samples
                        sim_count += 1

                pbar.update(pool_size)

        for team in all_teams:
            final_rank_probs[team][1] /= sim_count

        return {
            "final_rank": final_rank_probs,
            "rank_point": rank_point_dist,
            "major_contrib": major_contrib_ests
        }
