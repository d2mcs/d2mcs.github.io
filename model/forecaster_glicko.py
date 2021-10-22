"""This module contains the Glicko-2 based model used to compute team
ratings and win probabilities.
"""

import random
from math import pi, exp, log
import copy

class Glicko2Model:
    """Glicko-2 ratings model implemented following
    http://glicko.net/glicko/glicko2.pdf

    Parameters
    ----------
    tau : int
        System parameter tau
    """
    def __init__(self, tau):
        self.player_ratings = {}
        self.team_ratings = {}

        self._tau = tau
        self.k = 0 # not used but needed for compatibility

    def _get_player_rating(self, player_id):
        """Private method for getting or initializing a player's
        rating. This is used to estimate an initial team rating.
        """
        if player_id not in self.player_ratings:
            self.player_ratings[player_id] = 0
        return self.player_ratings[player_id]

    def _initialize_team(self, team, player_ids):
        """Private method for initializing a team's rating using the
        last recorded rating of each of its players.
        """
        player_ratings = [self._get_player_rating(pid) for pid in player_ids]
        team_rating = sum(player_ratings)/len(player_ratings)
        self.team_ratings[team] = (team_rating, 2, 0.06)

    def get_team_rating(self, team):
        """Obtain's a team's rating on the Glicko-1 scale.

        Parameters
        ----------
        team : hashable
            Intended to be an integer team ID but anything that can act
            as a dictionary key to identify the team is fine.

        Returns
        -------
        float
            Team rating on the Glicko-1 scale.
        """
        return self.team_ratings[team][0]*173.7178 + 1500

    def get_team_rating_tuple(self, team):
        """Obtain's a team's rating, rating deviation, and rating
        volatility on the Glicko-1 scale.

        Parameters
        ----------
        team : hashable
            Intended to be an integer team ID but anything that can act
            as a dictionary key to identify the team is fine.

        Returns
        -------
        float
            Team rating on the Glicko-1 scale.
        float
            Team rating deviation on the Glicko-1 scale.
        float
            Team rating volatility on the Glicko-1 scale.
        """
        return (self.team_ratings[team][0]*173.7178 + 1500,
                self.team_ratings[team][1]*173.7178,
                self.team_ratings[team][2])

    def get_win_prob(self, team1, team2, use_rd=True):
        """Computes the win probability for a single match.

        Parameters
        ----------
        team1 : hashable
            Dictionary key (intended to be intenger team id) for team 1
        team2 : hashable
            Dictionary key (intended to be intenger team id) for team 1
        use_rd : bool, default=True
            If true, rating deviation will be used in win probability
            estimation. This accounts for uncertainty in team rating
            estimates and will make predictions less confident.

        Returns
        -------
        float
            Probability (in the range [0, 1]) of team 1 winning.
        """
        phi_1 = self._phi(team1)
        phi_2 = self._phi(team2)
        mu_1 = self._mu(team1)
        mu_2 = self._mu(team2)
        if use_rd:
            phi = (phi_1**2 + phi_2**2)**(1/2)
            return 1/(1 + exp(-self._g(phi)*(mu_1 - mu_2)))
        else:
            return 1/(1 + exp(mu_2 - mu_1))

    def _g(self, phi):
        """http://glicko.net/glicko/glicko2.pdf"""
        return 1/((1 + 3*(phi**2)/(pi**2))**(1/2))

    def _E(self, mu, mu_j, phi_j):
        """http://glicko.net/glicko/glicko2.pdf"""
        return 1/(1 + exp(-self._g(phi_j)*(mu - mu_j)))

    def _mu(self, team):
        """Team rating, mu"""
        return self.team_ratings[team][0]

    def _phi(self, team):
        """Team rating deviation, phi"""
        return self.team_ratings[team][1]

    def _sigma(self, team):
        """Team volatility, sigma"""
        return self.team_ratings[team][2]

    def _vol_f(self, x, a, delta, phi, v):
        """f(x) function used for volatility calculation"""
        return (
            (exp(x)*(delta**2 - phi**2 - v - exp(x))
             / (2*( (phi**2 + v + exp(x))**2 )))
            - (x - a)/(self._tau**2)
        )

    def _get_new_sigma(self, sigma, delta, phi, v):
        """Iteratively computes the new volatility (sigma) value"""
        epsilon = 1e-06
        a = log(sigma**2)
        A = a
        if delta**2 > phi**2 + v:
            B = log(delta**2 - phi**2 - v)
        else:
            k = 1
            while self._vol_f(a - k*self._tau, a, delta, phi, v) < 0:
                k = k + 1
            B = a - k*self._tau

        f_A = self._vol_f(A, a, delta, phi, v)
        f_B = self._vol_f(B, a, delta, phi, v)
        while abs(B - A) > epsilon:
            C = A + (A - B)*f_A/(f_B - f_A)
            f_C = self._vol_f(C, a, delta, phi, v)

            if f_C*f_B < 0:
                A = B
                f_A = f_B
            else:
                f_A = f_A/2
            B = C
            f_B = f_C
        return exp(A/2)

    def update_ratings(self, team1, team2, score):
        """Updates model ratings given two teams and the results of a
        series between those teams. Note that this function should
        generally NOT be used for updating ratings, as Glicko-2 works
        best in batches. It only exists for compatibility with the Elo
        forecaster for simulations (Glicko explicitly accounts for
        rating uncertainty so even there it shouldn't be necessary).

        Parameters
        ----------
        team1 : list of int
            list of player IDs for team 1.
        team2 : list of int
            list of player IDs for team 2.
        score : tuple of (int, int)
            tuple containing number of team 1 wins and team 2 wins
        """
        match_count = sum(score)
        match_dict = {team1: [(team2, score[0]/match_count)],
                      team2: [(team1, score[1]/match_count)]}
        self.update_ratings_batch(match_dict)

    def update_ratings_batch(self, match_dict):
        """Updates model ratings using a large number of results. This
        is how the Glicko model should typically be updated.

        Parameters
        ----------
        match_dict : dict
            List of match results for a large number of teams. Results
            should be included both as A v. B and B v. A even though
            this is technically redundant. The dictionary should look
            something like this (for clarity I use A, B, and C as the
            teams but in practice these should be integer team IDs):
            {
                "A": [("B", 1), ("B", 0), ("C", 0)],
                "B": [("A", 0), ("A", 1), ("C", 1)],
                "C": [("A", 1), ("B", 0)]
            }
        """
        new_ratings = {}
        for team, matches in match_dict.items():
            mu = self._mu(team)
            phi = self._phi(team)

            v = 0
            for match in matches:
                phi_j = self._phi(match[0])
                mu_j = self._mu(match[0])
                E_j = self._E(mu, mu_j, phi_j)
                v += (self._g(phi_j)**2) * E_j * (1 - E_j)
            v = 1/v

            rating_update = 0
            for match in matches:
                phi_j = self._phi(match[0])
                mu_j = self._mu(match[0])
                score = match[1]
                rating_update+=self._g(phi_j)*(score - self._E(mu,mu_j,phi_j))
            delta = v*rating_update

            sigma_new = self._get_new_sigma(self._sigma(team), delta, phi, v)
            phi_star = (phi**2 + sigma_new**2)**(1/2)

            phi_new = 1/((1/(phi_star**2) + 1/v)**(1/2))
            mu_new = mu + phi_new**2 * rating_update
            new_ratings[team] = (mu_new, phi_new, sigma_new)

        for team, rating in new_ratings.items():
            self.team_ratings[team] = rating

    def compute_ratings(self, matches, stop_after=None):
        """Updates model ratings given an iterable containing match
        date (teams, winner/loser). This differs slightly from the
        indended Glicko-2 update method in the following ways:

        - Rating periods are not a standard length of time. Instead,
          updates are batched by league. Whenever no matches are played
          in a league for 5 days, all matches which haven't been
          processed yet are used to update ratings.
        - Partially as a consequence of the above, rating deviation
          does not increase over time for teams that have not competed
          in the rating period.

        Parameters
        ----------
        matches : iterable of Match
            Iterable containing Match objects (from match_data.py)
        stop_after : int, default=None
            If provided, the model will ignore matches played after
            this date (provided as a unix timestamp)

        Returns
        -------
        dict
            If track_history=True, this maps each unique team ID in the
            provided matches to a list of (team rating, timestamp)
            tuples tracking that team's rating over time where each
            rating is a float and each timestamp is an int (unix
            timestamp, in seconds). If track_history=False, an empty
            dictionary is returned instead.
        """
        league_matches = {}
        # the rating period can be longer than this, but if no matches
        # are played in a league in this timeframe that league will be
        # used for an update
        min_rating_period_duration = 60*60*24*5 # 5 days
        for match in matches:
            if stop_after is not None and match.timestamp > stop_after:
                break

            if match.radiant_id not in self.team_ratings:
                self._initialize_team(match.radiant_id, match.radiant)
            if match.dire_id not in self.team_ratings:
                self._initialize_team(match.dire_id, match.dire)

            radiant = match.radiant_id
            dire = match.dire_id
            if match.league_id in league_matches:
                league_matches[match.league_id]["last"] = match.timestamp
            else:
                league_matches[match.league_id] = {
                    "last" : match.timestamp, "matches": {}}

            batch = league_matches[match.league_id]["matches"]
            if radiant not in batch:
                batch[radiant] = [(dire, match.radiant_win)]
            else:
                batch[radiant].append((dire, match.radiant_win))
            if dire not in batch:
                batch[dire] = [(radiant, 1 - match.radiant_win)]
            else:
                batch[dire].append((radiant, 1 - match.radiant_win))

            for league in list(league_matches.keys()):
                if (match.timestamp - league_matches[league]["last"]
                      > min_rating_period_duration):
                    self.update_ratings_batch(
                        league_matches[league]["matches"])
                    del league_matches[league]

        for league in league_matches.keys():
            self.update_ratings_batch(league_matches[league]["matches"])

    def compute_ratings_evaluation_mode(self, matches, bins=20,
            start_at=None, stop_after=None, max_tier=3):
        """Function for calculating how well model estimations line up
        with actual outcomes. Matches are binned by estimated
        probability thresholds (e.g., win probability between 10% and
        12%) then the actually probability these events occur is
        calculated.

        Ratings are always updated using all league data up to the
        match metrics are being computed for. This makes computation
        a bit slow because the entire league batch has to be processed
        for each match.

        Parameters
        ----------
        matches : iterable of Match
            Iterable containing Match objects (from match_data.py)
        bins : int, default=20
            The number of probability thresholds to bin matches into
        start_at : int, default=0
            Unix timestamp representing time of earliest match to use
            for metric calculation.
        stop_after : int, default=2147483647
            Unix timestamp representing time of latest match to use for
            metric calculation.
        max_tier : int, default=3
            Maximum tier of match to consider in metric calculation.

        Returns
        -------
        list of tuple of (float, int)
            Actual probabilities of the outcomes forecast by the model.
            Estimated probabilities for each bin i are between (i/bins)
            and ((i + 1)/bins). Each tuple contains the probability of
            that outcome and the number of matches the probability was
            computed over.
        float
            Mean squared error over all matches
        """
        league_matches = {}
        # the rating period can be longer than this, but if no matches
        # are played in a league in this timeframe that league will be
        # used for an update
        min_rating_period_duration = 60*60*24*5

        count_bins = [[0, 0] for _ in range(bins)]
        model_sse = 0
        baseline_sse = 0
        match_count = 0
        for match in matches:
            if start_at is not None and match.timestamp < start_at:
                continue
            if stop_after is not None and match.timestamp > stop_after:
                break

            if match.radiant_id not in self.team_ratings:
                self._initialize_team(match.radiant_id, match.radiant)
            if match.dire_id not in self.team_ratings:
                self._initialize_team(match.dire_id, match.dire)

            radiant = match.radiant_id
            dire = match.dire_id
            if match.league_id in league_matches:
                league_matches[match.league_id]["last"] = match.timestamp
            else:
                league_matches[match.league_id] = {
                    "last" : match.timestamp, "matches": {}}

            batch = league_matches[match.league_id]["matches"]
            if radiant not in batch:
                batch[radiant] = [(dire, match.radiant_win)]
            else:
                batch[radiant].append((dire, match.radiant_win))
            if dire not in batch:
                batch[dire] = [(radiant, 1 - match.radiant_win)]
            else:
                batch[dire].append((radiant, 1 - match.radiant_win))

            if match.league_tier <= max_tier:
                # temporarily end ratings period to make predictions
                prev_ratings = {team: self.team_ratings[team]
                    for team in league_matches[match.league_id]["matches"]}
                self.update_ratings_batch(
                    league_matches[match.league_id]["matches"])
                win_p_r = self.get_win_prob(radiant, dire)

                count_bins[int(win_p_r*bins)][0] += match.radiant_win
                count_bins[int(win_p_r*bins)][1] += 1
                model_sse += pow(match.radiant_win - win_p_r, 2)
                baseline_sse += pow(match.radiant_win - 0.5, 2)
                match_count += 1
                for team, rating in prev_ratings.items():
                    self.team_ratings[team] = rating

            for league in list(league_matches.keys()):
                if (match.timestamp - league_matches[league]["last"]
                      > min_rating_period_duration):
                    self.update_ratings_batch(
                        league_matches[league]["matches"])
                    del league_matches[league]
        for league in league_matches.keys():
            self.update_ratings_batch(league_matches[league]["matches"])

        prob_bins = []
        for event_count, total_count in count_bins:
            if total_count > 0:
                prob_bins.append((event_count/total_count, total_count))
            else:
                prob_bins.append((0, 0))

        skill_score = 1 - (model_sse/match_count)/(baseline_sse/match_count)
        return prob_bins, skill_score
