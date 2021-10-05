"""This module contains code for generating HTML probability reports
using the TI simulator.
"""

from datetime import datetime
import argparse
import json

from jinja2 import Template

from simulator import TISimulator

def format_percentage(prob, n_samples):
    """Formats a float probability into something a little more
    readable.

    Floating point error means guaranteed probabilities may not be
    exactly 1 or 0 so instead the code uses thresholds that couldn't
    be passed unless the outcome always or never happened given the
    number of samples.
    """
    if prob < 1/n_samples:
        return "-"
    elif prob > (n_samples - 1)/n_samples:
        return "✓"
    elif prob < 0.001:
        return "<0.1%"
    elif prob > 0.999:
        return ">99.9%"
    else:
        return f"{prob*100:.1f}%"

def color_prob(p):
    """Calculates a shade of purple for a probability"""
    # generic purple color scaling
    return (hex(int(240 - 120*p))[2:].zfill(2)
          + hex(int(240 - 130*p))[2:].zfill(2)
          + hex(int(240 - 70*p))[2:].zfill(2))

def format_gs_rank_probabilities(rank_probs, n_samples):
    """Formats rank/bracket probabilities"""
    formatted_probs = {"a": {}, "b": {}}
    for group in ["a", "b"]:
        for team, rank_prob in sorted(rank_probs[group].items(),
                key=lambda x: -sum([i*p for i,p in enumerate(x[1])])):
            probs = {i: rank_prob[i] for i in range(len(rank_prob))}
            probs["upper"] = sum(rank_prob[:4])
            probs["lower"] = sum(rank_prob[4:8])
            probs["elim"] = rank_prob[8]

            formatted_probs[group][team] = {}
            for outcome, prob in probs.items():
                formatted_probs[group][team][outcome] = format_percentage(prob,
                    n_samples)
            formatted_probs[group][team]["upper-color"] = (
                hex(int(235 - 200*probs["upper"]))[2:].zfill(2) +
                hex(int(255 - 90*probs["upper"]))[2:].zfill(2) +
                hex(int(235 - 170*probs["upper"]))[2:].zfill(2))
            formatted_probs[group][team]["lower-color"] = (
                hex(int(255 - 40*probs["lower"]))[2:].zfill(2) +
                hex(int(230 - 145*probs["lower"]))[2:].zfill(2) +
                hex(int(205 - 200*probs["lower"]))[2:].zfill(2))
            formatted_probs[group][team]["elim-color"] = (
                hex(int(255 - 30*probs["elim"]))[2:].zfill(2) +
                hex(int(230 - 185*probs["elim"]))[2:].zfill(2) +
                hex(int(225 - 170*probs["elim"]))[2:].zfill(2))
            for i in range(len(rank_prob)):
                formatted_probs[group][team][f"{i}-color"]=color_prob(probs[i])

    return formatted_probs

def format_tiebreak_probabilities(tiebreak_probs, n_samples):
    """Formats tiebreaker probabilities"""
    formatted_probs = {"a": {}, "b": {}}
    for group in ["a", "b"]:
        for boundary, probs in tiebreak_probs[group].items():
            formatted_probs[group][boundary] = [{
                    "prob": format_percentage(prob, n_samples),
                    "color": color_prob(prob)}
                for prob in probs]
            overall_prob = sum(probs)
            formatted_probs[group][boundary].append({
                    "prob": format_percentage(overall_prob, n_samples),
                    "color": color_prob(overall_prob)})
    return formatted_probs

def format_match_probabilities(match_probs):
    """Formats and generates a color for a dict of probabilities"""
    colors = [color_prob(p) for p in match_probs]
    return [f"{p*100:.0f}%" for p in match_probs], colors

def format_final_rank_probabilities(rank_probs, n_samples):
    """Formats final rank probabilities"""
    formatted_probs = {}
    for team, prob_dict in sorted(rank_probs.items(), key=lambda x: x[1]["1"]):
        formatted_probs[team] = []
        for rank, prob in prob_dict.items():
            formatted_probs[team].append({
                    "prob": format_percentage(prob, n_samples),
                    "color": color_prob(prob)})

    return formatted_probs

def predict_matches(sim, matches, static_ratings):
    """Code for computing each team's record and the probabilities of
    them wining each match. Match probabilities are computed only with
    present information (e.g., a team's Day 4 rating is not used to
    compute Day 2 probabilities even on the Day 4 report).
    """
    records = {team: [0,0,0] for team in sim.rosters.keys()}
    formatted_matches = {"a": [], "b": []}
    for group in ["a", "b"]:
        for match_list in matches[group]:
            formatted_matches[group].append([])
            for match in match_list:
                match_probs = sim.model.get_bo2_probs(
                    sim._get_team(match[0]), sim._get_team(match[1]))
                if match[2] != -1:
                    # match has already been played, so update team
                    # ratings and current records
                    team1 = sim._get_team(match[0])
                    team2 = sim._get_team(match[1])
                    if not static_ratings:
                        result = (match[2], 2 - match[2])
                        sim.model.update_ratings(team1, team2, result)

                    records[match[0]][2 - match[2]] += 1
                    records[match[1]][match[2]] += 1

                formatted_probs, colors=format_match_probabilities(match_probs)
                formatted_matches[group][-1].append({
                    "teams": (match[0], match[1]), "result": match[2],
                    "probs": formatted_probs, "colors": colors})
    return records, formatted_matches

def generate_html(ratings_file, matches, output_file, n_samples, folder, k,
                  timestamp="", static_ratings=False, tabs=None,
                  title="The International 10"):
    """Generates an output report with group stage probabilities.

    Parameters
    ----------
    ratings_file : str
        Path to JSON file mapping team names to ratings. The file
        should look something like this:
        {
          "Team A": 1500,
          "Team B": 1600
        }
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
    output_file : str
        Name of output file to save.
    n_samples : int
        Number of Monte Carlo samples to simulate.
    folder : str
        Folder name to save output to / look for data from.
    timestamp : str, default=""
        Timestamp to put at the top of the report. This is a parameter
        instead of being determined from the current time because I
        generate multiple reports and want them to have the same
        timestamp
    static_ratings : bool, default=False
        If true, ratings will not be updated over the course of a
        simulation. This is used for the fixed-ratings output.
    tabs : dict, default=None
        Dictionary containing the name and suffix of each tab. See
        main() or retroactive_predictions() for an example. If not
        provided, the tab dropdown will not show up on the output
        report.
    title : str, default="The International 10"
        Name to put at the top of the report.
    """
    sim = TISimulator.from_ratings_file(ratings_file, k,
        static_ratings=static_ratings)
    with open(f"data/{folder}/groups.json") as group_f:
        groups = json.load(group_f)

    group_rank_probs, tiebreak_probs, final_rank_probs = sim.sim_group_stage(
        groups, matches, n_samples)

    records, formatted_matches = predict_matches(sim, matches, static_ratings)
    ratings = {team: f"{sim.model.get_team_rating(sim._get_team(team)):.0f}"
              for team in sim.rosters.keys()}
    formatted_gs_probs = format_gs_rank_probabilities(group_rank_probs,
        n_samples)
    formatted_tie_probs = format_tiebreak_probabilities(
        tiebreak_probs, n_samples)
    formatted_final_probs = format_final_rank_probabilities(
        final_rank_probs, n_samples)

    with open("data/template.html") as input_f:
        template_str = input_f.read()
    template = Template(template_str)

    output = template.render(gs_prob_strs=formatted_gs_probs,
        final_prob_strs=formatted_final_probs,
        tiebreak_prob_strs=formatted_tie_probs, records=records,
        ratings=ratings, matches=formatted_matches, timestamp=timestamp,
        n_samples=n_samples, output_file=output_file, tabs=tabs, title=title)

    if tabs is not None:
        with open(f"{folder}/{output_file}{tabs['active'][1]}",
              "w") as output_f:
            output_f.write(output)
    else:
        with open(f"{folder}/{output_file}", "w") as output_f:
            output_f.write(output)

def generate_team_ratings(max_tier, k, p, folder, stop_after=None):
    """Code for generating rating estimates for each provided team
    roster. Will only work if matches.db exists in the data folder.
    """
    sim = TISimulator.from_match_data(f"data/{folder}/rosters.json",
        "data/matches.db", max_tier, k, p, stop_after=stop_after)
    sim.save_ratings(f"data/{folder}/elo_ratings.json")

def retroactive_predictions(timestamp, k, n_samples, tournament, train_elo):
    """Code for generating retroactive TI predictions. Will only work
    if matches.db exists in the data folder.
    """
    tabs = {
        "all": [["Pre-tournament", "-pre.html"], ["Day 1", "-1.html"],
                ["Day 2", "-2.html"], ["Day 3", "-3.html"],
                ["Current", ".html"]]
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
        generate_team_ratings(3, k, 1.5, tournament, stop_after)

    for i, tab in enumerate(tabs["all"]):
        tabs["active"] = tab
        with open(f"data/{tournament}/matches.json") as match_f:
            matches = json.load(match_f)
        for group in ["a", "b"]:
            for day in range(i, 4):
                for match in matches[group][day]:
                    match[2] = -1

        generate_html(f"data/{tournament}/elo_ratings.json", matches, "elo",
            n_samples, tournament, k, timestamp, tabs=tabs, title=title)
        generate_html(f"data/{tournament}/fixed_ratings.json", matches,
            "fixed", n_samples, tournament, k, timestamp, static_ratings=True,
            tabs=tabs, title=title)

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
    parser.add_argument("-s","--static_ratings", action='store_false',
        default=True, help="Disables static ratings, meaning team ratings will"
        " be updated based on match results.")
    parser.add_argument("-e","--train-elo", action='store_true', default=False,
        help="Retrain Elo model before generating probabilities. Will only "
        "work if matches.db and ti_rosters.json exist in the data folder")
    parser.add_argument("-r","--retroactive-predict", action='store_true',
        default=False, help="Generates retroactive predictions for past TIs. "
        "Will only work if matches.db and ti_rosters.json exist in the data "
        "folder")
    parser.add_argument("-f","--full-report", action='store_true',
        default=False, help="Generates a full report for use on the website. "
        "Will only work if matches.db and ti_rosters.json exist in the data "
        "folder")
    parser.add_argument("-k", default=55, type=int, help="k parameter for the "
        "Elo model.")
    args = parser.parse_args()

    k = args.k
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    n_samples = args.n_samples

    if args.retroactive_predict:
        for event in ["ti7", "ti8", "ti9"]:
            retroactive_predictions(timestamp, k, n_samples,
                                    event, args.train_elo)
    elif args.full_report:
        if args.train_elo:
            generate_team_ratings(3, k, 1.5, "ti10")

        tabs = {
            "active": ["Current", ".html"],
            "all": [["Current", ".html"]]
        }
        with open("data/ti10/matches.json") as match_f:
            matches = json.load(match_f)
        generate_html("data/ti10/elo_ratings.json", matches, "elo", n_samples,
                      "ti10", k, timestamp, tabs=tabs)
        generate_html("data/ti10/fixed_ratings.json", matches, "fixed",
                      n_samples, "ti10", k, timestamp,
                      static_ratings=True, tabs=tabs)
    else:
        with open("data/ti10/matches.json") as match_f:
            matches = json.load(match_f)
        if validate_ti10_files():
            generate_html("data/ti10/elo_ratings.json", matches, "output.html",
                          n_samples, "ti10", k, timestamp,
                          static_ratings=args.static_ratings)
            print("Output saved to ti10/output.html")

if __name__ == "__main__":
    main()
