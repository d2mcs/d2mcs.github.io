"""This module contains the command-line interface for generating
probability reports using the TI simulator.
"""

from datetime import datetime
import argparse
import json
import sys

from model.forecaster import PlayerModel, TeamModel
from model.forecaster_glicko import Glicko2Model
from model.match_data import MatchDatabase
from website.generate_html import generate_html_ti

def generate_team_ratings_elo(max_tier, k, p, folder, stop_after=None):
    """Code for generating rating estimates for each provided team
    roster. Will only work if matches.db and a list of team rosters
    exist in the data folder.
    """
    match_db = MatchDatabase("data/matches.db")
    player_ids = match_db.get_player_ids()
    p_model = PlayerModel(player_ids, k, p)
    p_model.compute_ratings(match_db.get_matches(max_tier),
        stop_after=stop_after)

    with open(f"data/{folder}/rosters.json") as roster_f:
        rosters = json.load(roster_f)
    model = TeamModel.from_player_model(p_model, rosters)

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

    with open(f"data/{folder}/team_ids.json") as tid_f:
        team_ids = json.load(tid_f)

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

def retroactive_predictions(timestamp, k, n_samples, tournament,
                            train_elo, use_cached):
    """Code for generating retroactive TI predictions. Will only work
    if matches.db exists in the data folder.
    """
    tabs = {
        "all": [["Pre-tournament", "-pre.html"], ["Day 1", "-1.html"],
                ["Day 2", "-2.html"], ["Day 3", "-3.html"],
                ["Day 4", "-4.html"], ["Current", ".html"]]
    }
    if tournament == "ti10":
        stop_after = datetime.fromisoformat("2021-10-05").timestamp()
        title = "The International 10"
    elif tournament == "ti9":
        stop_after = datetime.fromisoformat("2019-08-13").timestamp()
        title = "The International 2019"
    elif tournament == "ti8":
        stop_after = datetime.fromisoformat("2018-08-13").timestamp()
        title = "The International 2018"
    elif tournament == "ti7":
        stop_after = datetime.fromisoformat("2017-07-31").timestamp()
        title = "The International 2017"
    else:
        raise ValueError("Invalid tournament")

    if train_elo:
        generate_team_ratings_elo(3, k, 1.5, tournament, stop_after)
        generate_team_ratings_glicko(3, 0.5, tournament, stop_after)

    for i, tab in enumerate(tabs["all"]):
        tabs["active"] = tab
        with open(f"data/{tournament}/matches.json") as match_f:
            matches = json.load(match_f)
        for group in ["a", "b"]:
            for day in range(i, 4):
                for match in matches[group][day]:
                    match[2] = -1

        generate_html_ti(f"data/{tournament}/elo_ratings.json", matches, "elo",
            n_samples, tournament, k, timestamp, tabs=tabs, title=title,
            use_cached=use_cached,
            bracket_file=f"data/{tournament}/main_event_matches.json"
                         if tab[0] == "Current" else None)
        generate_html_ti(f"data/{tournament}/fixed_ratings.json", matches,
            "fixed", n_samples, tournament, k, timestamp, static_ratings=True,
            tabs=tabs, title=title, use_cached=use_cached,
            bracket_file=f"data/{tournament}/main_event_matches.json"
                         if tab[0] == "Current" else None)

def validate_ti10_files():
    """Some simple checks for the ti10 data files to help users catch
    errors before they become python exceptions
    """
    data = {}
    for file in ["elo_ratings", "groups", "matches"]:
        try:
            with open(f"data/ti10/{file}.json") as json_f:
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
    parser.add_argument("-c","--use-cached", action='store_true',
        default=False,
        help=argparse.SUPPRESS if "-H" not in sys.argv else "Use cached "
        "probability estimates if they exist.")
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
    use_cached = args.use_cached

    if args.retroactive_predict:
        for event in ["ti7", "ti8", "ti9", "ti10"]:
            retroactive_predictions(timestamp, k, n_samples,
                                    event, args.train_elo, use_cached)
    elif args.full_report:
        if args.train_elo:
            generate_team_ratings_elo(3, k, 1.5, "ti10",
                stop_after=datetime.fromisoformat("2021-10-05").timestamp())

        tabs = {
            "active": ["Current", ".html"],
            "all": [["Pre-tournament", "-pre.html"], ["Day 1", "-1.html"],
                    ["Day 2", "-2.html"], ["Day 3", "-3.html"],
                    ["Current", ".html"]]
        }
        with open("data/ti10/matches.json") as match_f:
            matches = json.load(match_f)
        generate_html_ti("data/ti10/elo_ratings.json", matches, "elo",
                         n_samples, "ti10", k, timestamp, tabs=tabs,
                         bracket_file="data/ti10/main_event_matches.json",
                         use_cached=use_cached)
        generate_html_ti("data/ti10/fixed_ratings.json", matches, "fixed",
                         n_samples, "ti10", k, timestamp,
                         static_ratings=True, tabs=tabs,
                         bracket_file="data/ti10/main_event_matches.json",
                         use_cached=use_cached)
    else:
        with open("data/ti10/matches.json") as match_f:
            matches = json.load(match_f)
        if validate_ti10_files():
            generate_html_ti("data/ti10/elo_ratings.json", matches,
                             "output.html", n_samples, "ti10", k, timestamp,
                             static_ratings=args.static_ratings,
                             use_cached=use_cached)
            print("Output saved to ti10/output.html")

if __name__ == "__main__":
    main()
