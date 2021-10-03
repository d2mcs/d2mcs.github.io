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
CREATE TABLE teams(name TEXT, id INT PRIMARY KEY);
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
        self.league_id = result_tuple[15]
        self.league_tier = result_tuple[16]

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

    def get_matches(self, max_tier=3):
        """Collects all matches from the database with a tier less than
        or equal to max_tier. Tiers are between 1 (premier) and 7
        (show match). For more information, see:
        liquipedia.net/dota2/Portal:Tournaments

        Parameters
        ----------
        max_tier: int, default=3
            The highest numeric tier of matches to collect

        Yields
        ------
        Match
            Match object containing match information
        """
        match_query = f"""SELECT radiant_acc1, radiant_acc2, radiant_acc3,
                radiant_acc4, radiant_acc5, dire_acc1, dire_acc2, dire_acc3,
                dire_acc4, dire_acc5, radiant_win, radiant_teamid, dire_teamid,
                timestamp, match_id, matches.league_id, liquipediatier.tier
            FROM matches JOIN liquipediatier
                         ON matches.league_id = liquipediatier.league_id
            WHERE liquipediatier.tier <= {max_tier}
            ORDER BY match_id"""

        for row in self.cur.execute(match_query):
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
