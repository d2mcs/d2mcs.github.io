"""This module contains the command-line interface for generating
probability reports using the TI and DPC league simulators.
"""

from datetime import datetime, date
import argparse
import json
import sys
import http.server
import webbrowser
import os

from model.forecaster import PlayerModel, TeamModel
from model.forecaster_glicko import Glicko2Model
from model.match_data import MatchDatabase
from website.report import (generate_data_ti, generate_html_ti,
                            generate_data_dpc, generate_html_dpc,
                            generate_html_global_rankings)

def generate_team_ratings_elo(max_tier, k, p, folder, stop_after=None):
    """Code for generating rating estimates for each provided team
    roster. Will only work if matches.db and a list of team rosters
    exist in the data folder.
    """
    match_db = MatchDatabase("data/matches.db")
    player_ids = match_db.get_player_ids()
    id_to_region = match_db.get_id_region_map()
    p_model = PlayerModel(player_ids, k, p, tid_region_map=id_to_region)
    p_model.compute_ratings(match_db.get_matches(max_tier),
        stop_after=stop_after)

    with open(f"data/{folder}/team_data.json") as tid_f:
        regions = {team: data["region"]
                   for team, data in json.load(tid_f).items()}

    with open(f"data/{folder}/rosters.json") as roster_f:
        rosters = json.load(roster_f)
    model = TeamModel.from_player_model(p_model, rosters, regions)

    with open(f"data/{folder}/elo_ratings.json", "w") as output_f:
        output_f.write("{\n")
        for i, team in enumerate(rosters.keys()):
            rating = model.get_team_rating(team)
            if i != len(rosters) - 1:
                output_f.write(f'  "{team}": {rating:.2f},\n')
            else:
                output_f.write(f'  "{team}": {rating:.2f}\n')
        output_f.write("}\n")

def generate_team_ratings_glicko(max_tier, tau, folder, stop_after=None):
    """Code for generating rating estimates for each provided team
    roster. Will only work if matches.db exists in the data folder.
    """
    match_db = MatchDatabase("data/matches.db")
    model = Glicko2Model(tau)
    model.compute_ratings(match_db.get_matches(max_tier),stop_after=stop_after)

    with open(f"data/{folder}/team_data.json") as tid_f:
        team_ids = {team: data["id"] for team,data in json.load(tid_f).items()}

    with open(f"data/{folder}/glicko_ratings.json", "w") as output_f:
        output_f.write("{\n")
        for i, (team, tid) in enumerate(team_ids.items()):
            mean, rd, sigma = model.get_team_rating_tuple(tid)
            rating = f"[{mean:.2f}, {rd:.4f}, {sigma:.7f}]"
            if i != len(team_ids) - 1:
                output_f.write(f'  "{team}": {rating},\n')
            else:
                output_f.write(f'  "{team}": {rating}\n')
        output_f.write("}\n")

def generate_global_ratings_elo(max_tier, k, p, dpc_season, timestamp):
    """Code for generating an Elo rating for every team which competed
    in the provided DPC season. Uses the last recorded rating for the
    teamid of that team.
    """
    match_db = MatchDatabase("data/matches.db")
    player_ids = match_db.get_player_ids()
    id_to_region = match_db.get_id_region_map()
    p_model = PlayerModel(player_ids, k, p, tid_region_map=id_to_region)
    rating_history = p_model.compute_ratings(match_db.get_matches(max_tier),
                                             track_history=True)

    team_ids = {}
    regions = {}
    for region in ["na", "sa", "weu", "eeu", "cn", "sea"]:
        with open(f"data/dpc/{dpc_season}/{region}/team_data.json") as roster_f:
            for team, team_data in json.load(roster_f).items():
                team_ids[team] = team_data["id"]
                regions[team] = region
    output_data = {
        "timestamp": timestamp,
        "ratings": [(team, regions[team],
                    float(f"{rating_history[tid][-1][0]:.2f}"),
                    str(date.fromtimestamp(rating_history[tid][-1][1])))
                    for team, tid in team_ids.items()]
    }

    with open(f"data/global/elo_ratings.json", "w") as output_f:
        json.dump(output_data, output_f)

def retroactive_ti_predictions(timestamp, k, n_samples, tournament,
                               train_elo, html_only=False):
    """Code for generating retroactive TI predictions. Will only work
    if matches.db exists in the data folder.
    """
    if tournament in ["ti/8", "ti/9"]:
        tabs = [["Aug. 25 (Current)", ""],["Aug 18 (Group Stage Day 4)", "-4"],
                ["Aug. 17 (Group Stage Day 3)", "-3"],
                ["Aug. 16 (Group Stage Day 2)", "-2"],
                ["Aug. 15 (Group Stage Day 1)", "-1"],
                ["Aug. 14 (Pre-tournament)", "-pre"]]

    if tournament == "ti/10":
        stop_after = datetime.fromisoformat("2021-10-05").timestamp()
        title = "The International 10"
        tabs = [["Oct. 17 (Current)", ""],["Oct 10 (Group Stage Day 4)", "-4"],
                ["Oct. 9 (Group Stage Day 3)", "-3"],
                ["Oct. 8 (Group Stage Day 2)", "-2"],
                ["Oct. 7 (Group Stage Day 1)", "-1"],
                ["Oct. 6 (Pre-tournament)", "-pre"]]
    elif tournament == "ti/9":
        stop_after = datetime.fromisoformat("2019-08-13").timestamp()
        title = "The International 2019"
    elif tournament == "ti/8":
        stop_after = datetime.fromisoformat("2018-08-13").timestamp()
        title = "The International 2018"
    elif tournament == "ti/7":
        stop_after = datetime.fromisoformat("2017-07-31").timestamp()
        title = "The International 2017"
        tabs = [["Aug. 12 (Current)", ""],["Aug 5 (Group Stage Day 4)", "-4"],
                ["Aug. 4 (Group Stage Day 3)", "-3"],
                ["Aug. 3 (Group Stage Day 2)", "-2"],
                ["Aug. 2 (Group Stage Day 1)", "-1"],
                ["Aug. 1 (Pre-tournament)", "-pre"]]
    else:
        raise ValueError("Invalid tournament")

    generate_html_ti(tournament + "/forecast.html", tabs, title)
    if html_only:
        return

    if train_elo:
        generate_team_ratings_elo(3, k, 1.5, tournament, stop_after)
        generate_team_ratings_glicko(3, 0.5, tournament, stop_after)

    for i, tab in enumerate(reversed(tabs)):
        with open(f"data/{tournament}/matches.json") as match_f:
            matches = json.load(match_f)
        for group in ["a", "b"]:
            for day in range(i, 4):
                for match in matches[group][day]:
                    match[2] = -1

        generate_data_ti(f"data/{tournament}/elo_ratings.json", matches,
            "elo" + tab[1], n_samples, tournament, k, timestamp,
            bracket_file=f"data/{tournament}/main_event_matches.json"
                         if tab[1] == "" else None)
        generate_data_ti(f"data/{tournament}/fixed_ratings.json", matches,
            "fixed" + tab[1], n_samples, tournament, k, timestamp,
            static_ratings=True,
            bracket_file=f"data/{tournament}/main_event_matches.json"
                         if tab[1] == "" else None)

def retroactive_dpc_predictions(timestamp, k, n_samples, region,
                                train_elo, html_only=False):
    """Code for generating retroactive DPC league predictions. Will
    only work if matches.db exists in the data folder.
    """
    tabs = [["May 23 (Current)", ""],
            ["May 21 (Week 6)", "-6"], ["May 16 (Week 5)", "-5"],
            ["May 9 (Week 4)", "-4"], ["May 2 (Week 3)", "-3"],
            ["Apr. 25 (Week 2)", "-2"], ["Apr. 18 (Week 1)", "-1"],
            ["Apr. 11 (Pre-tournament)", "-pre"]]
    full_name = {
        "na": "North America", "sa": "South America", "weu": "Western Europe",
        "eeu": "Eastern Europe", "cn": "China", "sea": "Southeast Asia"
    }
    wildcard_slots = {"sea": 1, "eeu": 1, "cn": 2, "weu": 2, "na": 0, "sa": 0}

    if train_elo:
        generate_team_ratings_elo(3, k, 1.5, "dpc/sp21/"+region,
            stop_after=datetime.fromisoformat("2021-04-10").timestamp())
    generate_html_dpc(f"dpc/sp21/{region}/forecast.html", tabs,
        "DPC Spring 2021: " + full_name[region], wildcard_slots[region])
    if html_only:
        return

    for i, tab in enumerate(reversed(tabs)):
        with open(f"data/dpc/sp21/{region}/matches.json") as match_f:
            matches = json.load(match_f)
        for division in ["upper", "lower"]:
            for day in range(i, 6):
                for match in matches[division][day]:
                    match[2] = []
            if tab[1] != "":
                matches["tiebreak"][division] = {}

        generate_data_dpc(f"data/dpc/sp21/{region}/elo_ratings.json",
                          matches, "elo" + tab[1], n_samples,
                          f"dpc/sp21/{region}", k, wildcard_slots[region],
                          timestamp=timestamp, static_ratings=False)
        generate_data_dpc(f"data/dpc/sp21/{region}/fixed_ratings.json",
                          matches, "fixed" + tab[1], n_samples,
                          f"dpc/sp21/{region}", k, wildcard_slots[region],
                          timestamp=timestamp, static_ratings=True)


def validate_ti10_files():
    """Some simple checks for the ti10 data files to help users catch
    errors before they become python exceptions
    """
    data = {}
    for file in ["elo_ratings", "groups", "matches"]:
        try:
            with open(f"data/ti/10/{file}.json") as json_f:
                data[file] = json.load(json_f)
        except json.decoder.JSONDecodeError:
            print(f"ERROR: Failed to load {file}.json: invalid JSON")
            return False
    teams = set()
    for team, rating in data["elo_ratings"].items():
        teams.add(team)
        if not isinstance(rating, (float, int)):
            print("ERROR: team ratings in elo_ratings.json must be numbers")
            return False
    for group, group_teams in data["groups"].items():
        for team in group_teams:
            if team not in teams:
                print(f"ERROR: {team} (groups.json) is "
                      "not in elo_ratings.json")
                return False
    for group in ["a", "b"]:
        if len(data["matches"][group]) != 4:
            print("ERROR: matches must be split into 4 lists "
                  "(one for each match day)")
            return False
        for match_list in data["matches"][group]:
            for match in match_list:
                for team in [match[0], match[1]]:
                    if team not in data["groups"][group]:
                        print(f"ERROR: match {match} in group {group} contains"
                              f" team '{team}' which is not in group {group} "
                              "in groups.json")
                        return False
                if match[2] not in [0, 1, 2, -1]:
                    print("ERROR: match results must be one of [0, 1, 2, -1]")
                    return False
    return True

def main():
    """Command-line interface for probability report generation"""
    parser = argparse.ArgumentParser(
        description="Generate probability report for TI10 group stage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("n_samples", default=100000, type=int,
        help="Number of Monte Carlo samples to simulate")
    parser.add_argument("-H", action='store_true', default=False,
        help="Detailed help: shows options that are only useful if you have "
        "the non-public matches.db database. These are used for updating pages"
        " on d2mcs.github.io.")
    parser.add_argument("-s","--static_ratings", action='store_false',
        default=True, help="Disables static ratings, meaning team ratings will"
        " be updated based on match results.")
    parser.add_argument("-e","--train-elo", action='store_true', default=False,
        help=argparse.SUPPRESS if "-H" not in sys.argv else "Retrain Elo model"
        " before generating probabilities.")
    parser.add_argument("-r","--retroactive-predict", action='store_true',
        default=False,
        help=argparse.SUPPRESS if "-H" not in sys.argv else "Generates "
        "retroactive predictions for past TIs.")
    parser.add_argument("--rd", action='store_true', default=False,
        help=argparse.SUPPRESS if "-H" not in sys.argv else "Generates "
        "retroactive predictions for past DPC leagues.")
    parser.add_argument("-g","--global_ratings", action='store_true',
        default=False,
        help=argparse.SUPPRESS if "-H" not in sys.argv else "Generates "
        "a global rating of all 96 DPC teams")
    parser.add_argument("--html", action='store_true', default=False,
        help=argparse.SUPPRESS if "-H" not in sys.argv else "Updates HTML "
        "files without generating new predictions.")
    parser.add_argument("-f","--full-report", action='store_true',
        default=False, help="Generates a full report for use on the website.")
    parser.add_argument("-k", default=55, type=int, help="k parameter for the "
        "Elo model.")
    if "-H" in sys.argv:
        parser.print_help()
        exit()
    args = parser.parse_args()

    k = args.k
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    n_samples = args.n_samples
    tabs = [["Oct. 17 (Current)", ""],["Oct 10 (Group Stage Day 4)", "-4"],
            ["Oct. 9 (Group Stage Day 3)", "-3"],
            ["Oct. 8 (Group Stage Day 2)", "-2"],
            ["Oct. 7 (Group Stage Day 1)", "-1"],
            ["Oct. 6 (Pre-tournament)", "-pre"]]

    if args.retroactive_predict:
        for event in ["ti/7", "ti/8", "ti/9", "ti/10"]:
            retroactive_ti_predictions(timestamp, k, n_samples,
                                       event, args.train_elo, args.html)
    elif args.rd:
        for region in ["sea", "eeu", "cn", "weu", "na", "sa"]:
            retroactive_dpc_predictions(timestamp, k, n_samples,
                                        region, args.train_elo, args.html)
    elif args.full_report:
        if args.train_elo:
            generate_team_ratings_elo(3, k, 1.5, "ti/10",
                stop_after=datetime.fromisoformat("2021-10-05").timestamp())

        with open("data/ti/10/matches.json") as match_f:
            matches = json.load(match_f)

        generate_html_ti("ti/10/forecast.html", tabs, "The International 10")
        if not args.html:
            generate_data_ti("data/ti/10/elo_ratings.json", matches, "elo",
                n_samples, "ti/10", k, timestamp,
                bracket_file="data/ti/10/main_event_matches.json")
            generate_data_ti("data/ti/10/fixed_ratings.json", matches, "fixed",
                n_samples, "ti/10", k, timestamp, static_ratings=True,
                bracket_file="data/ti/10/main_event_matches.json")
    elif args.global_ratings:
        if args.train_elo:
            generate_global_ratings_elo(3, k, 1.5, "sp21", timestamp)
        generate_html_global_rankings("global_ratings.html", "sp21")
    else:
        with open("data/ti/10/matches.json") as match_f:
            matches = json.load(match_f)
        if validate_ti10_files():
            generate_html_ti("ti/10/user_forecast.html", [["Current", ""]],
                             "The International 10")
            generate_data_ti("data/ti/10/elo_ratings.json", matches,
                             "custom", n_samples, "ti/10", k, timestamp,
                             static_ratings=args.static_ratings)
            os.chdir("..")
            print("Output running at http://localhost:8000/ti/10/"
                  "user_forecast.html?model=custom. Press ctrl+c"
                  " or close this window to exit.")
            webbrowser.open(
                "http://localhost:8000/ti/10/user_forecast.html?model=custom")
            server = http.server.HTTPServer(('127.0.0.1', 8000),
                http.server.SimpleHTTPRequestHandler)
            server.serve_forever()

if __name__ == "__main__":
    main()
