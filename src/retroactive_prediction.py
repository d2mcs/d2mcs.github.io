"""This module contains the command-line interface for generating
retroactive forecasts of past events. This is mostly used for sanity
checking model updates but is usable without the match database as long
as the -e parameter isn't used.
"""

from datetime import datetime
import argparse
import json
from pathlib import Path
import os

from website.report import (generate_data_ti, generate_html_ti,
    generate_data_dpc, generate_html_dpc,
    generate_data_major, generate_html_major, generate_team_ratings_elo)

def retroactive_ti_predictions(timestamp, k, n_samples, tournament,
                               train_elo, html_only=False):
    """Code for generating retroactive TI predictions."""
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
        generate_team_ratings_elo(2, k, 1.5, tournament, stop_after)

    for i, tab in enumerate(reversed(tabs)):
        with open(f"data/{tournament}/matches.json") as match_f:
            matches = json.load(match_f)
        for group in ["a", "b"]:
            for day in range(i, 4):
                for match in matches[group][day]:
                    match[2] = -1
            if tab[1] != "" and "tiebreak" in matches:
                matches["tiebreak"][group] = {}

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
    """Code for generating retroactive DPC league predictions."""
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
        generate_team_ratings_elo(2, k, 1.5, "dpc/sp21/"+region,
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

def retroactive_major_predictions(timestamp, k, n_samples,
                                  train_elo, html_only=False):
    """TODO
    HTML report for major events hasn't been finished yet. This is
    currently only used for testing purposes
    """
    stop_after = datetime.fromisoformat("2021-06-01").timestamp()
    title = "Animajor"
    tabs = [["Current", ""]]

    if train_elo:
        generate_team_ratings_elo(2, k, 1.5, "dpc/sp21/major", stop_after)

    with open("data/dpc/sp21/major/matches.json") as match_f:
        matches = json.load(match_f)

    generate_data_major("data/dpc/sp21/major/elo_ratings.json", matches,
            "elo" + tabs[0][1], n_samples, "dpc/sp21/major", k, timestamp)

def main():
    """Command-line interface for retroactive report generation"""
    # code must be run from the src/ folder
    os.chdir(str(Path(__file__).parent))

    parser = argparse.ArgumentParser()
    parser.add_argument("n_samples", default=100000, type=int,
        help="Number of Monte Carlo samples to simulate")
    parser.add_argument("-e","--train-elo", action='store_true', default=False,
        help="Retrain Elo model before generating probabilities.")
    parser.add_argument("-k", default=55, type=int, help="k parameter for the "
        "Elo model.")
    parser.add_argument("--html", action='store_true', default=False,
        help="Updates HTML files without generating new predictions.")
    parser.add_argument("-t", "--ti", action='store_true', default=False,
        help="Generate retroactive TI predictions")
    parser.add_argument("-d", "--dpc", action='store_true', default=False,
        help="Generate retroactive DPC league predictions")
    parser.add_argument("-m", "--major", action='store_true', default=False,
        help="Generate retroactive DPC major predictions")

    args = parser.parse_args()

    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")

    if args.ti:
        for event in ["ti/7", "ti/8", "ti/9", "ti/10"]:
            retroactive_ti_predictions(timestamp, args.k, args.n_samples,
                                       event, args.train_elo, args.html)
    if args.dpc:
        for region in ["sea", "eeu", "cn", "weu", "na", "sa"]:
            retroactive_dpc_predictions(timestamp, args.k, args.n_samples,
                                        region, args.train_elo, args.html)
    if args.major:
        retroactive_major_predictions(timestamp, args.k, args.n_samples,
                                      args.train_elo, args.html)

if __name__ == "__main__":
    main()
