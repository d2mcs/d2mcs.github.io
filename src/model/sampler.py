"""This module contains the Monte-Carlo samplers used for computing
probability estimates. Multiprocessing is used to speed up computation
of results.
"""

from multiprocessing import Pool
import json
import copy

from tqdm import tqdm

from model.simulator import TIEliminationBracket, TIGroupStage
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

    def get_bo2_probs(self, team1, team2, draw_adjustment=False):
        """Computes win/draw/loss probabilities for a bo2 match.

        Parameters
        ----------
        team1 : hashable
            Team 1 identifier (string team name or integer team ID)
        team2 : hashable
            Team 2 identifier (string team name or integer team ID)
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
        win_p_t1 = self.model.get_win_prob(team1, team2)
        if not draw_adjustment:
            p2_0 = pow(win_p_t1, 2)
            p0_2 = pow(1 - win_p_t1, 2)
        else:
            # winner probability adjustment shouldn't exceed 95%
            p2_0 = win_p_t1*min(max(0.95, win_p_t1), win_p_t1 + 0.1)
            p0_2 = (1-win_p_t1)*min(max(0.95, 1-win_p_t1), (1-win_p_t1) + 0.1)
        return (p2_0, (1 - (p2_0 + p0_2)), p0_2)

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
        return cls(model, static_ratings=static_ratings)

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
            model.ratings[team] = ((rating_tuple[0] - 1500)/173.7178,
                rating_tuple[1]/173.7178, rating_tuple[2])
        return cls(model, static_ratings=static_ratings)

class TISampler(Sampler):
    """TI Simulator. For each Monte Carlo sample a GroupStage object
    and an EliminationBracket object are created using a working copy
    of the prediction model. Multiprocessing is then used to collect
    many samples reasonably quickly.

    Parameters
    ----------
    Identical to Simulator. See above for details
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
            (See: sim_group_stage documentation below)
        matches : dict
            (See: sim_group_stage documentation below)
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

        points_a, tiebreak_sizes_a = gs_sim.simulate(groups["a"], matches["a"])
        points_b, tiebreak_sizes_b = gs_sim.simulate(groups["b"], matches["b"])

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
            (See: sim_group_stage documentation below)
        bracket : dict
            Bracket with team seeds and match results (results
            currently are not used)
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

    def sample_main_event(self, groups, matches, bracket, n_trials):
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
        return (group_rank_probs, tiebreak_probs,
                final_rank_probs, record_probs, point_rank_probs)
