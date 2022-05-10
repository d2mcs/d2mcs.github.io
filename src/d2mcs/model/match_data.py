"""This module acts as a wrapper for the (non-public, unfortunately)
SQLlite match database. For running the Elo model you will need a
matches table and a leagues table with the following schema:

CREATE TABLE matches(
    league_id INT, match_id INT PRIMARY KEY, timestamp INT,
    radiant_win BOOLEAN,
    radiant_teamid INT, radiant_acc1 INT, radiant_acc2 INT,
    radiant_acc3 INT, radiant_acc4 INT, radiant_acc5 INT
    dire_teamid INT, dire_acc1 INT, dire_acc2 INT, dire_acc3 INT,
    dire_acc4 INT, dire_acc5 INT);

(Additional columns are fine and you don't need to have the same
primary key but you do need all the listed columns)

League, Player and Team ID tables use the following schema:

CREATE TABLE liquipediatier(league_id INT PRIMARY KEY, tier INT);
CREATE TABLE players(name TEXT, id INT PRIMARY KEY);
CREATE TABLE teams(name TEXT, id INT PRIMARY KEY, region TEXT);
"""

import sqlite3

class Match:
    """Class for organizing the results of a database query into a more
    usable format. Takes a result tuple from sqlite3 as input to
    populate the relevant member variables
    """
    def __init__(self, result_tuple):
        self.radiant = result_tuple[:5]
        self.dire = result_tuple[5:10]
        self.radiant_win = result_tuple[10]
        self.radiant_id = result_tuple[11]
        self.dire_id = result_tuple[12]
        self.timestamp = result_tuple[13]
        self.match_id = result_tuple[14]
        self.series_id = result_tuple[15]
        self.league_id = result_tuple[16]
        self.league_tier = result_tuple[17]

class MatchDatabase:
    """Class for obtaining useful information (match info, players,
    teams) from the match database

    Parameters
    ----------
    database_file : str
        Path to sqlite database
    """
    def __init__(self, database_file):
        self.con = sqlite3.connect(database_file)
        self.cur = self.con.cursor()

    def get_matches(self, max_tier=3, min_pool=100000000):
        """Collects all matches from the database with a tier less than
        or equal to max_tier. Tiers are between 1 (premier) and 7
        (show match). For more information, see:
        liquipedia.net/dota2/Portal:Tournaments

        Parameters
        ----------
        max_tier : int, default=3
            The highest numeric tier of matches to collect
        min_pool : int, default=100000000
            If the tournament has a prize pool larger than this value,
            it will be included regardless of liquipedia tier.

        Yields
        ------
        Match
            Match object containing match information
        """
        match_query = f"""SELECT radiant_acc1, radiant_acc2, radiant_acc3,
                radiant_acc4, radiant_acc5, dire_acc1, dire_acc2, dire_acc3,
                dire_acc4, dire_acc5, radiant_win, radiant_teamid, dire_teamid,
                timestamp, match_id, series_id, matches.league_id,
                liquipediatier.tier,
                radiant_ban1, radiant_ban2, dire_ban1, dire_ban2
            FROM matches JOIN liquipediatier
                         ON matches.league_id = liquipediatier.league_id
            WHERE liquipediatier.tier <= {max_tier}
               OR liquipediatier.prizepool >= {min_pool}
            ORDER BY match_id"""

        for row in self.cur.execute(match_query):
            # if first 4 bans are empty, something has gone wrong
            # (this happens for some matches abandonded due
            #  to lobby issues)
            if not all(row[18:]) == 0:
                yield Match(row)

    def get_player_ids(self, min_count=0):
        """Collects a list of players who have competed in a minium
        number of competitive matches. Returns the player IDs of these
        players.

        Parameters
        ----------
        min_count: int, default=0
            Only players which appear in this number of competitive
            matches will be returned.

        Returns
        -------
        set of int
            set of player IDs
        """
        acc_query = f"""SELECT acc FROM
            (
                SELECT radiant_acc1 as acc FROM matches UNION ALL
                SELECT radiant_acc2 as acc FROM matches UNION ALL
                SELECT radiant_acc3 as acc FROM matches UNION ALL
                SELECT radiant_acc4 as acc FROM matches UNION ALL
                SELECT radiant_acc5 as acc FROM matches UNION ALL
                SELECT dire_acc1 as acc FROM matches UNION ALL
                SELECT dire_acc2 as acc FROM matches UNION ALL
                SELECT dire_acc3 as acc FROM matches UNION ALL
                SELECT dire_acc4 as acc FROM matches UNION ALL
                SELECT dire_acc5 as acc FROM matches
            )
            GROUP BY acc HAVING count(acc) >= {min_count}"""

        player_ids = set()
        for pid in self.cur.execute(acc_query):
            player_ids.add(pid[0])
        return player_ids

    def get_id_player_map(self):
        """Creates a dictionary mapping player IDs to player names
        using data pulled from Liquipedia.

        Returns
        -------
        dict of int to str
            Mapping from player ID to username
        """
        id_to_player = {}
        for row in self.cur.execute("SELECT id, name FROM players"):
            id_to_player[row[0]] = row[1]
        return id_to_player

    def get_id_team_map(self):
        """Creates a dictionary mapping team IDs to team names using
        data pulled from Liquipedia. This information is also available
        from the steam API but I didn't collect it when I was
        downloading match data.

        Returns
        -------
        dict of int to str
            Mapping from team ID to team name
        """
        id_to_team = {}
        for row in self.cur.execute("SELECT id, name FROM teams"):
            id_to_team[row[0]] = row[1]
        return id_to_team

    def get_id_region_map(self):
        """Creates a dictionary mapping team IDs to DPC regions using
        data pulled from Liquipedia.

        Returns
        -------
        dict of int to str
            Mapping from team ID to team name
        """
        id_to_region = {}
        for row in self.cur.execute("SELECT id, region FROM teams"):
            id_to_region[row[0]] = row[1]
        return id_to_region

    def predict_match_setting(self):
        """Predicts whether each match in the dataset was played online
        or on LAN using the assumption that LAN event should contain
        matches between teams of at least 4/6 DPC regions within a three
        day window. Note that matches between teams of the same region
        are ignored for this count (qualifiers are frequently played on
        the same league ticket on the same day, but should not be
        detected as LAN).

        This is obviously not a perfect method -- not all LAN events
        include teams from 4+ regions and if qualifiers are played on
        the same league ticket with an NA/SA match and WEU/EEU match
        within 3 days they will be categorized as LAN matches -- but it
        works fine for most cases, particularly after WEU/EEU and NA/SA
        were separated.

        Returns
        -------
        dict
            Mapping from match ID (int) to one of {"lan", "online"}
        """
        id_to_region = self.get_id_region_map()

        match_query = """SELECT radiant_teamid, dire_teamid,
                                league_id, match_id, timestamp
                         FROM matches"""
        match_setting_map = {}
        match_window = {}

        def _predict_region(league_id):
            """small nested helper function for predicting regions"""
            regions = set()
            for region1, region2 in match_window[league_id]["regions"]:
                if region1 == "UNK" or region2 == "UNK":
                    continue
                if region1 != region2:
                    # only add regions if they're different
                    regions.update({region1, region2})
            if len(regions) > 3:
                guess = "lan"
            else:
                guess = "online"
            # if there are 4+ unique regions in the window assume that
            # every match in the window was a LAN match
            for match_id in match_window[league_id]["matches"]:
                match_setting_map[match_id] = guess
            del match_window[league_id] # clear window

        for row in self.cur.execute(match_query):
            radiant_id, dire_id, league_id, match_id, timestamp = row
            if (league_id in match_window and match_window[league_id
                    ]["timestamp"] - timestamp > 3*24*60*60):
                # if the first match in the league window is more than
                # 3 days old, guess whether the window is online/lan
                _predict_region(league_id)
            if league_id not in match_window:
                # no window for league, create one
                match_window[league_id] = {
                    "timestamp": timestamp,
                    "regions": [(id_to_region.get(radiant_id, "UNK"),
                                 id_to_region.get(dire_id, "UNK"))],
                    "matches": [match_id]
                }
            else:
                # add match to current window
                match_window[league_id]["regions"].append(
                    (id_to_region.get(radiant_id, "UNK"),
                     id_to_region.get(dire_id, "UNK")))
                match_window[league_id]["matches"].append(match_id)

        for league_id in list(match_window.keys()):
            # empty any remaining windows
            _predict_region(league_id)
        return match_setting_map
