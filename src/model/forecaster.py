"""This module contains the Elo-based model used to compute team ratings
and win probabilities.
"""

import random

class PlayerModel:
    """Modified Elo model. Unlike typical Elo models, ratings are
    tracked by player rather than by team to account for roster
    instability and substitutes. Team ratings are simply the average
    rating of the team's players.

    Player ratings are updated unevenly after each game such that all
    players on the same team will eventually have the same rating (if
    the team wins, the players with the lowest ratings will receive the
    largest rating boost. If the team loses, the players with the
    highest ratings will receive the highest rating drop).

    Unless team rosters change during a tournament it is best to
    convert this to a team model using TeamModel.from_player_model()
    before running a Monte-Carlo sim to avoid the added overhead from
    player tracking.

    Parameters
    ----------
    players : int
        List of player IDs to track. Because the model is player
        -oriented, team IDs aren't needed.
    k : int
        The usual k paramter used by any Elo system. Increasing k
        increases the rate at which ratings are adjusted.
    p : float
        This controls how much the model pushes players on the same
        team to have the same rating. With p=0, every player on a team
        receives an identical rating adjustment after each game. As p
        increases the model becomes more aggressive about matching
        player ratings, making the model more similar to a traditional
        team-based Elo model.
    league_p : float, default=0.75
        This controls how much league quality (deterined using
        Liquipedia tier) controls rating adjustments. Rating
        adjustments are multiplied by:

            tier^(-league_p)

        This means that as league_p -> 0, league quality is ignored and
        as league_p -> infinity, only tier 1 events are considered
        (Anything higher than ~3 will make tier 2+ events near
        worthless for ratings)
    tid_region_map : dict, default=None
        Maps team IDs to DPC regions. If provided, the model will
        maintain a rating for each region based on how well teams in
        that region perform against teams from other regions.
    region_share : float, default=0.1
        If region data is used, this controls the percentage of each
        rating update after an international match which is used to
        update the region rather than the team.
    """
    def __init__(self, players, k, p, league_p=0.75,
                 tid_region_map=None, region_share=0.1):
        self.region_ratings = {"NA": 0, "SA": 0, "WEU": 0,
                               "EEU": 0, "CN": 0, "SEA": 0}
        self.ratings = self._initialize_ratings(players)
        self.k = k
        self.p = p
        self.league_p = league_p
        self.tid_to_region = tid_region_map
        self.region_share = region_share

    def _region_rating(self, region):
        """Private accessor function for region_ratings. This makes it
        a bit easier to control the default value for when the region
        is unknown. Currently the default rating for an unknown region
        is the minimum region rating.
        """
        if region in self.region_ratings:
            return self.region_ratings[region]
        else:
            return min(self.region_ratings.values())

    def _initialize_ratings(self, player_ids):
        """Private method for giving players an initial rating"""
        ratings = {}
        for player_id in player_ids:
            ratings[player_id] = 1500
        return ratings

    def _get_modifier_distribution(self, team, win):
        """Private method for computing how the rating update should be
        distributed among players. When a team loses, each player's
        share of the rating loss is computed as:

            player share = player rating / team rating

        When a team wins, each player's share of the rating loss is
        computed as:

            player share = (1 / player rating) / sum(1 / player rating)
                                                   over all players

        A power factor (p) above 1 will exaggerate the differences
        between player rating, making the model push players towards
        having the same rating more quickly

        Parameters
        ----------
        team: list of int
            list of player IDs for the team
        win: bool
            True if the team won, False otherwise

        Returns
        -------
        list of int
            5-element list containing each player's proportion of the
            rating update. Should always sum to 1.
        """
        if win:
            inverse_sum = sum([pow(1/self.ratings[pid], self.p)
                               for pid in team])
            player_mod = [pow(1/self.ratings[pid],self.p)/inverse_sum
                          for pid in team]
        else:
            rating_sum = sum([pow(self.ratings[id], self.p) for id in team])
            player_mod=[pow(self.ratings[id],self.p)/rating_sum for id in team]

        return player_mod

    def get_team_rating(self, player_ids, region="UNK"):
        """Computes the team rating given a list of player IDs.

        Parameters
        ----------
        player_ids : list of int
            List of player IDs to compute a rating for. Throws an error
            if the list length isn't 5 rather than just computing the
            average anyway because teams should always have 5 players
        region : str, default=None
            If provided, the region's modifier will be applied to the
            team's rating.

        Returns
        -------
        float
            Team rating.
        """
        if len(player_ids) != 5:
            raise ValueError("Teams must have 5 players")
        team_rating = sum([self.ratings[id] for id in player_ids])/5
        if region is not None:
            team_rating += self._region_rating(region.upper())
        return team_rating

    def get_win_prob(self, team1, team2, regions=None):
        """Computes the win probability for a single match.

        Parameters
        ----------
        team1 : list of int
            list of player IDs for team 1.
        team2 : list of int
            list of player IDs for team 2.
        regions : tuple of (str, str), default=None
            If provided, win probabilities will use the teams' regional
            modifiers.

        Returns
        -------
        float
            Probability (in the range [0, 1]) of team 1 winning.
        """
        if regions is None:
            rating_t1 = self.get_team_rating(team1)
            rating_t2 = self.get_team_rating(team2)
        else:
            rating_t1 = self.get_team_rating(team1, regions[0])
            rating_t2 = self.get_team_rating(team2, regions[1])
        win_p_t1 = 1/(1 + pow(10, (rating_t2 - rating_t1)/400))
        return win_p_t1

    def update_ratings(self, team1, team2, score, league_tier=None,
            regions=None):
        """Updates model ratings given two teams and the results of a
        series between those teams.

        Parameters
        ----------
        team1 : list of int
            list of player IDs for team 1.
        team2 : list of int
            list of player IDs for team 2.
        score : tuple of (int, int)
            tuple containing number of team 1 wins and team 2 wins
        league_tier : int [1 - 7]
            Tier of tournament match is from. If provided, will reduce
            the adjustment size for lower-tier tournaments
        regions : tuple of (str, str), default=None
            If provided, win probabilities will use the teams' regional
            modifiers.
        """
        if regions is None:
            team1_rating = self.get_team_rating(team1)
            team2_rating = self.get_team_rating(team2)
        else:
            team1_rating = self.get_team_rating(team1, regions[0])
            team2_rating = self.get_team_rating(team2, regions[1])
        quality_mod1 = 1.5 - min(1, max(0, (team1_rating - 1000)/1000))
        quality_mod2 = 1.5 - min(1, max(0, (team2_rating - 1000)/1000))
        if league_tier is not None:
            league_mod = pow(league_tier, -self.league_p)
        else:
            league_mod = 1

        expected_score = self.get_win_prob(team1, team2, regions)
        actual_score = score[0]/sum(score)
        adjustment = league_mod*self.k*(actual_score - expected_score)

        delta_t1 = adjustment*quality_mod1
        delta_t2 = -adjustment*quality_mod2
        if regions is not None and regions[0] != regions[1]:
            if regions[0] != "UNK":
                self.region_ratings[regions[0]] += delta_t1*self.region_share
                delta_t1 = delta_t1*(1 - self.region_share)
            if regions[1] != "UNK":
                self.region_ratings[regions[1]] += delta_t2*self.region_share
                delta_t2 = delta_t2*(1 - self.region_share)

        player_dist1 = self._get_modifier_distribution(team1,
            actual_score > expected_score)
        player_dist2 = self._get_modifier_distribution(team2,
            actual_score < expected_score)
        for i in range(5):
            self.ratings[team1[i]] += delta_t1*5*player_dist1[i]
            self.ratings[team2[i]] += delta_t2*5*player_dist2[i]

    def _get_series(self, matches, stop_after=None):
        """Private helper method for compute_ratings which collects all
        series in the order they occured.
        """
        series = {}
        tid_counts = {}
        for match in matches:
            if stop_after is not None and match.timestamp > stop_after:
                break
            series_id = match.series_id
            if series_id in series:
                # series IDs are occasionally reused for reasons I
                # don't entirely understand.
                while (series_id in series and (
                        match.radiant_id not in series[series_id]
                        or match.dire_id not in series[series_id])):
                    series_id += .1

            if series_id in series:
                radiant_win = match.radiant_win
                series[series_id][match.radiant_id]["score"] += radiant_win
                series[series_id][match.dire_id]["score"] += 1 - radiant_win
            else:
                if series_id == 0:
                    series_id = random.random()
                if match.radiant_id == match.dire_id:
                    continue
                series[series_id] = {
                    match.radiant_id: {
                        "score": match.radiant_win, "players": match.radiant},
                    match.dire_id: {
                        "score": 1 - match.radiant_win, "players": match.dire},
                    "timestamp": match.timestamp,
                    "league_tier": match.league_tier
                }
            tid_counts[match.radiant_id] = tid_counts.get(match.radiant_id,0)+1
            tid_counts[match.dire_id] = tid_counts.get(match.dire_id,0)+1
        ordered_series = []
        for series in sorted(series.values(), key=lambda x: x['timestamp']):
            teams = [k for k in series.keys() if isinstance(k, (int, float))]
            if tid_counts[teams[0]] < 10 or tid_counts[teams[0]] < 10:
                continue
            players = [series[t]["players"] for t in teams]
            score = (series[teams[0]]["score"], series[teams[1]]["score"])
            ordered_series.append((teams, players, score,
                series["league_tier"], series["timestamp"]))
        return ordered_series

    def _get_regions(self, team1_id, team2_id):
        """Private helper method for compute_ratings which determines
        the region of two teams. If one team's region is unknown and
        the other's isn't, the unknown team's region is assumed to be
        the same as the known team's region.
        """
        region1 = self.tid_to_region.get(team1_id, "UNK")
        region2 = self.tid_to_region.get(team2_id, "UNK")
        if region1 == "UNK" and region2 != "UNK":
            region1 = region2
            self.tid_to_region[team1_id] = region2
        elif region2 == "UNK" and region1 != "UNK":
            region2 = region1
            self.tid_to_region[team2_id] = region1
        return region1, region2

    def compute_ratings(self, matches, track_history=False, stop_after=None):
        """Updates model ratings given an iterable containing match
        date (teams, winner/loser)

        Parameters
        ----------
        matches : iterable of Match
            Iterable containing Match objects (from match_data.py)
        track_history : bool, default=False
            If true, a running history of team ratings over time is
            maintained for all unique team IDs
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
        team_ratings = {}
        if self.tid_to_region is not None:
            for region in self.region_ratings:
                team_ratings["REGION/" + region] = []

        series_list = self._get_series(matches, stop_after)
        for series in series_list:
            team1_id, team2_id = series[0]
            team1_players, team2_players = series[1]
            score = series[2]
            league_tier = series[3]
            timestamp = series[4]

            if self.tid_to_region is not None:
                regions = self._get_regions(team1_id, team2_id)
            else:
                regions = None
            self.update_ratings(team1_players, team2_players,
                                score, league_tier, regions)

            if track_history:
                if team1_id not in team_ratings:
                    team_ratings[team1_id] = []
                if team2_id not in team_ratings:
                    team_ratings[team2_id] = []
                team_ratings[team1_id].append((self.get_team_rating(
                    team1_players, regions[0] if self.tid_to_region is not None
                                   else None), timestamp))
                team_ratings[team2_id].append((self.get_team_rating(
                    team2_players, regions[1] if self.tid_to_region is not None
                                   else None), timestamp))
                if self.tid_to_region is not None:
                    for region, rating in self.region_ratings.items():
                        team_ratings["REGION/" + region].append(
                            (rating, timestamp))


        return team_ratings

    def compute_ratings_evaluation_mode(self, matches, bins=20, start_at=0,
            stop_after=2147483647, max_tier=3, min_appearances=0):
        """Function for calculating how well model estimations line up
        with actual outcomes. Matches are binned by estimated
        probability thresholds (e.g., win probability between 10% and
        12%) then the actually probability these events occur is
        calculated.

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
        min_appearances : int, default=0
            Metrics will only be computed for teams which have played
            in this number of series.

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
        count_bins = [[0, 0] for _ in range(bins)]
        model_sse = 0
        baseline_sse = 0
        match_count = 0
        series_list = self._get_series(matches, stop_after)

        appearances = {}
        for series in series_list:
            team1_id, team2_id = series[0]
            team1_players, team2_players = series[1]
            score = series[2]
            league_tier = series[3]
            timestamp = series[4]

            for tid in (team1_id, team2_id):
                appearances[tid] = appearances.get(tid, 0) + 1

            if self.tid_to_region is not None:
                regions = self._get_regions(team1_id, team2_id)
            else:
                regions = None

            if timestamp > stop_after:
                break
            if timestamp > start_at and league_tier <= max_tier:
                if (appearances[team1_id] > min_appearances
                      and appearances[team2_id] > min_appearances):
                    win_p_t1 = self.get_win_prob(team1_players, team2_players,
                        regions if self.tid_to_region is not None else None)
                    actual_score = score[0]/sum(score)

                    count_bins[int(win_p_t1*bins)][0] += actual_score
                    count_bins[int(win_p_t1*bins)][1] += 1

                    model_sse += pow(actual_score - win_p_t1, 2)
                    baseline_sse += pow(actual_score - 0.5, 2)
                    match_count += 1

            self.update_ratings(team1_players, team2_players,
                                score, league_tier, regions)

        prob_bins = []
        for event_count, total_count in count_bins:
            if total_count > 0:
                prob_bins.append((event_count/total_count, total_count))
            else:
                prob_bins.append((0, 0))

        skill_score = 1 - (model_sse/match_count)/(baseline_sse/match_count)
        return prob_bins, skill_score

class TeamModel:
    """Team-based Elo model. There is not currently any code for
    training one, but player-based models can be converted to team-
    based ones to improve efficiency if player tracking isn't needed
    (this is usually the case for simulating an individual tournament,
    because team rosters remain consistent for the duration).

    Parameters
    ----------
    ratings: dict
        Maps team names to ratings. The dict should look something
        like this:
        {
          "Team A": 1500,
          "Team B": 1600
        }
    k: int
        The usual k paramter used by any Elo system. Increasing k
        increases the rate at which ratings are adjusted.
    """
    def __init__(self, ratings, k):
        self.ratings = ratings
        self.k = k

    @classmethod
    def from_player_model(cls, player_model, rosters, regions=None):
        """Generates a team-based model from a player-based one.

        Parameters
        ----------
        player_model : PlayerModel
            The PlayerModel to derive ratings from.
        rosters : dict
            Maps team names to player IDs. The dict should look something
            like this:
            {
              "Team A": [1, 2, 3, 4, 5],
              "Team B": [6, 7, 8, 9, 10]
            }
        regions : dict, default=None
            If provided, regional modifiers will be used when obtaining
            team ratings from the PlayerModel.
        Returns
        -------
        TeamModel
            Constructed TeamModel object
        """
        ratings = {team: player_model.get_team_rating(roster,
                       regions[team] if regions is not None else None)
                   for team, roster in rosters.items()}
        return cls(ratings, player_model.k)

    def get_team_rating(self, team):
        """Glorified dictionary access. Exists only for consistency
        with PlayerModel.

        Parameters
        ----------
        team : str
            Team name

        Returns
        -------
        float
            Team rating.
        """
        return self.ratings[team]

    def get_win_prob(self, team1, team2):
        """Computes the win probability for a single match.

        Parameters
        ----------
        team1 : str
            Team 1 name
        team2 : str
            Team 2 name

        Returns
        -------
        float
            Probability (in the range [0, 1]) of team 1 winning.
        """
        rating_t1 = self.get_team_rating(team1)
        rating_t2 = self.get_team_rating(team2)
        win_p_t1 = 1/(1 + pow(10, (rating_t2 - rating_t1)/400))
        return win_p_t1

    def update_ratings(self, team1, team2, score):
        """Updates model ratings given two teams and a the results of s
        series between those teams.

        Parameters
        ----------
        team1 : str
            Team 1 name
        team2 : str
            Team 2 name
        score : tuple of (int, int)
            tuple containing number of team 1 wins and team 2 wins
        """
        team1_rating = self.get_team_rating(team1)
        team2_rating = self.get_team_rating(team2)
        quality_mod1 = 1.5 - min(1, max(0, (team1_rating - 1000)/1000))
        quality_mod2 = 1.5 - min(1, max(0, (team2_rating - 1000)/1000))

        expected_score = self.get_win_prob(team1, team2)
        actual_score = score[0]/sum(score)
        adjustment_t1 = self.k*(actual_score - expected_score)

        self.ratings[team1] += adjustment_t1*quality_mod1
        self.ratings[team2] -= adjustment_t1*quality_mod2
