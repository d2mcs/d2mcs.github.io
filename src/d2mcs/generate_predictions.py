"""This module contains the command-line interface for generating
a forecast using current match results and team ratings.
"""

from datetime import datetime
import argparse
import json
import sys
from pathlib import Path
import http.server
import webbrowser
import os

from website.report import (generate_data_ti, generate_html_ti,
    generate_data_dpc, generate_html_dpc, generate_html_global_rankings,
    generate_team_ratings_elo, generate_global_ratings_elo,
    generate_html_dpc_major, generate_data_dpc_major)
from website.conf import EventConf

def validate_data_files(folder, groups, match_list_len):
    """Some simple checks for the data files to help users catch
    errors before they become python exceptions
    """
    data = {}
    for file in ["fixed_ratings", "teams", "matches"]:
        try:
            with open(f"data/{folder}/{file}.json") as json_f:
                data[file] = json.load(json_f)
        except json.decoder.JSONDecodeError:
            print(f"ERROR: Failed to load {file}.json: invalid JSON")
            return False
    teams = set()
    for team, rating in data["fixed_ratings"].items():
        teams.add(team)
        if not isinstance(rating, (float, int)):
            print("ERROR: team ratings in fixed_ratings.json must be numbers")
            return False
    for group, group_teams in data["teams"].items():
        for team in group_teams:
            if team not in teams:
                print(f"ERROR: {team} (teams.json) is not in "
                      "fixed_ratings.json")
                return False
    for group in groups:
        if len(data["matches"][group]) != match_list_len:
            print(f"ERROR: matches must be split into {match_list_len} lists "
                  "(one for each match day/week)")
            return False
        for match_list in data["matches"][group]:
            for match in match_list:
                for team in [match[0], match[1]]:
                    if team not in data["teams"][group]:
                        print(f"ERROR: match {match} in group {group} contains"
                              f" team '{team}' which is not in group {group} "
                              "in teams.json")
                        return False
                if match[2] not in [0, 1, 2, -1, [2,0],[2,1],[1,2],[0,2],
                                    ["W","-"], ["-", "W"], ["-", "-"], []]:
                    print("ERROR: match results must be one of [-1, 0, 1, 2]"
                          " (bo2)\n       or [[], [2,0], [2,1], etc.] "
                          "(bo1, bo3, bo5)")
                    return False
    return True

def generate_report(event, k, n_samples, timestamp, train_elo, html_only):
    """Generates a forecast with both Elo ratings and fixed ratings for
    use on the website.
    """
    config = EventConf(f"data/conf/{event}.json")
    tabs = config.tabs
    event_name = config.name

    if event == "ti10":
        if train_elo:
            generate_team_ratings_elo(3, k, 1.5, "ti/10")

        with open("data/ti/10/matches.json") as match_f:
            matches = json.load(match_f)
        generate_html_ti("ti/10/forecast.html", tabs, event_name)
        if not html_only:
            generate_data_ti("data/ti/10/elo_ratings_lan.json", matches,
                "elo", n_samples, "ti/10", k, timestamp,
                             bracket_file="data/ti/10/main_event_matches.json")
            generate_data_ti("data/ti/10/fixed_ratings.json", matches,
                "fixed", n_samples, "ti/10", k, timestamp,
                             static_ratings=True,
                             bracket_file="data/ti/10/main_event_matches.json")
    elif event[:3] == "dpc":
        tour = event.split("-")[-1]
        full_name = {
            "na": "North America", "sa": "South America",
            "weu": "Western Europe", "eeu": "Eastern Europe",
            "cn": "China", "sea": "Southeast Asia"
        }
        wildcard_slots = {"sea": 1, "eeu": 1, "cn": 2,
                          "weu": 2, "na": 0, "sa": 0}
        for region in ["na", "sa", "weu", "cn", "sea"]:
            if train_elo:
                generate_team_ratings_elo(3, k, 1.5, f"dpc/{tour}/{region}")
            with open(f"data/dpc/{tour}/{region}/matches.json") as match_f:
                matches = json.load(match_f)
            generate_html_dpc(f"dpc/{tour}/{region}/forecast.html", tabs,
                              f"{event_name}: {full_name[region]}",
                              wildcard_slots[region], matches)
            if html_only:
                return

            generate_data_dpc(
                f"data/dpc/{tour}/{region}/elo_ratings_online.json",
                 matches, "elo", n_samples, f"dpc/{tour}/{region}", k,
                 wildcard_slots[region], timestamp=timestamp,
                 static_ratings=False)
            generate_data_dpc(f"data/dpc/{tour}/{region}/fixed_ratings.json",
                              matches, "fixed", n_samples,
                              f"dpc/{tour}/{region}", k,wildcard_slots[region],
                              timestamp=timestamp, static_ratings=True)
    else:
        tour = event.split("-")[-1]
        if train_elo:
            generate_team_ratings_elo(3, k, 1.5, f"dpc/{tour}/major")

        with open(f"data/dpc/{tour}/major/matches.json") as match_f:
            matches = json.load(match_f)
        generate_html_dpc_major(f"dpc/{tour}/major/forecast.html",
                                tabs, event_name)

        if not html_only:
            generate_data_dpc_major(f"data/dpc/{tour}/major/elo_ratings_lan.json",
                matches, "elo", n_samples, f"dpc/{tour}/major", k, timestamp)
            generate_data_dpc_major(f"data/dpc/{tour}/major/fixed_ratings.json",
                matches, "fixed", n_samples, f"dpc/{tour}/major", k, timestamp,
                static_ratings=True)


def custom_report(event, region, k, n_samples, timestamp, static_ratings):
    """Generates a custom forecast with user-modified Elo ratings."""
    if event == "ti10":
        folder = "ti/10"
        with open("data/ti/10/matches.json") as match_f:
            matches = json.load(match_f)
        if validate_data_files("ti/10", ["a", "b"], 4):
            generate_data_ti("data/ti/10/fixed_ratings.json", matches,
                             "custom", n_samples, "ti/10", k, timestamp,
                             static_ratings=static_ratings)
        else:
            return
    else:
        tour = event.split("-")[-1]
        folder = f"dpc/{tour}/{region}"
        wildcard_slots = {"sea": 1, "eeu": 1, "cn": 2,
                          "weu": 2, "na": 0, "sa": 0}

        with open(f"data/dpc/{tour}/{region}/matches.json") as match_f:
            matches = json.load(match_f)
        if validate_data_files(f"dpc/{tour}/{region}", ["upper","lower"], 6):
            generate_data_dpc(f"data/dpc/{tour}/{region}/fixed_ratings.json",
                              matches, "custom", n_samples,
                              f"dpc/{tour}/{region}", k,
                              wildcard_slots[region], timestamp=timestamp,
                              static_ratings=static_ratings)
        else:
            return
    os.chdir("../..")
    print(f"Output running at http://localhost:8000/{folder}/forecast.html"
          "?model=custom. Press ctrl+c or close this window to exit.")
    webbrowser.open(
        f"http://localhost:8000/{folder}/forecast.html?model=custom")
    server = http.server.HTTPServer(('127.0.0.1', 8000),
        http.server.SimpleHTTPRequestHandler)
    server.serve_forever()

def main():
    """Command-line interface for probability report generation"""
    # code must be run from the d2mcs/ folder
    os.chdir(str(Path(__file__).parent))

    parser = argparse.ArgumentParser(
        description="Generate probability report for TI10 group stage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("event", type=str,
        choices=["ti10", "dpc-sp21", "dpc-wn21", "dpc-sp22", "maj-sp22"],
        help="Which event to generate predictions for.")
    parser.add_argument("n_samples", default=100000, type=int,
        help="Number of Monte Carlo samples to simulate")
    parser.add_argument("-H", action='store_true', default=False,
        help="Detailed help: shows options that are only useful if you have "
        "the non-public matches.db database. These are used for updating pages"
        " on d2mcs.github.io.")
    parser.add_argument("-r","--region", choices=["na", "sa", "weu", "eeu",
        "cn", "sea"], help="DPC region to generate predictions for")
    parser.add_argument("-s","--static_ratings", action='store_false',
        default=True, help="Disables static ratings, meaning team ratings will"
        " be updated based on match results.")
    parser.add_argument("-e","--train-elo", action='store_true', default=False,
        help=argparse.SUPPRESS if "-H" not in sys.argv else "Retrain Elo model"
        " before generating probabilities.")
    parser.add_argument("-g","--global_ratings", action='store_true',
        default=False,
        help=argparse.SUPPRESS if "-H" not in sys.argv else "Generates "
        "a global rating of all 96 DPC teams")
    parser.add_argument("--html", action='store_true', default=False,
        help=argparse.SUPPRESS if "-H" not in sys.argv else "Updates HTML "
        "files without generating new predictions.")
    parser.add_argument("-f","--full-report", action='store_true',
        default=False, help=argparse.SUPPRESS if "-H" not in sys.argv else "Ge"
        "nerates a full report for use on the website.")
    parser.add_argument("-k", default=35, type=int, help="k parameter for the "
        "Elo model. Unless the Elo model is being retrained this only matters "
        "if static ratings are disabled")
    if "-H" in sys.argv:
        parser.print_help()
        sys.exit()
    args = parser.parse_args()

    if args.event[:3]=="dpc" and not args.full_report and args.region is None:
        print("ERROR: region must be selected with -r for "
              "custom DPC league forecasts")
        sys.exit()

    k = args.k
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    n_samples = args.n_samples

    if args.full_report:
        generate_report(args.event, k, n_samples,
                        timestamp, args.train_elo, args.html)
    elif args.global_ratings:
        if args.event[:3] != "dpc":
            raise ValueError("Global ratings require a DPC season selection")
        tour = args.event[4:]
        if args.train_elo:
            generate_global_ratings_elo(3, k, 1.5, tour, timestamp)
        generate_html_global_rankings("global_ratings.html",tour, n_samples, k)
    else:
        custom_report(args.event, args.region, k, n_samples,
                      timestamp, args.static_ratings)

if __name__ == "__main__":
    main()
