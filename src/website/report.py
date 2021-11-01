"""This module contains code for generating an html report using
simulator predictions.
"""

import json
from pathlib import Path

from jinja2 import Template

from model.sampler import TISampler, DPCSampler

def get_gs_ranks(rank_probs, groups):
    """Orders group stage rank probabilities into a list and trims
    decimal places.
    """
    ranks = {group: [] for group in groups}
    for group in groups:
        for team, rank_prob in sorted(rank_probs[group].items(),
                key=lambda x: sum([i*p for i,p in enumerate(x[1])])):
            ranks[group].append({"team": team,
                "probs": [float(f"{p:.8f}") for p in rank_prob]})
    return ranks

def get_final_ranks(rank_probs):
    """Orders final rank probabilities into a list and trims
    decimal places.
    """
    ranks = []
    for team, prob_dict in sorted(rank_probs.items(),
            key=lambda x: -sum([i*p for i,p in enumerate(x[1].values())])):
        ranks.append({"team": team,
            "probs": [float(f"{p:.8f}") for p in prob_dict.values()]})
    return ranks

def get_records(record_probs, point_rank_probs, groups):
    """Orders record and expected rank per points probabilities into a
    list and trims decimal places.
    """
    records = {group: [] for group in groups}
    for group in groups:
        for team, record_prob in sorted(record_probs[group].items(),
                key=lambda x: -sum([i*p for i,p in enumerate(x[1])])):
            records[group].append({"team": team,
                "record_probs": [float(f"{p:.8f}") for p in record_prob],
                "point_rank_probs": [
                    [float(f"{p:.8f}") for p in probs.values()]
                    for probs in point_rank_probs[group][team].values()]})
    return records

def predict_matches_ti(sampler, matches, static_ratings):
    """Code for computing each team's record and the probabilities of
    them wining each match. Match probabilities are computed only with
    present information (e.g., a team's Day 4 rating is not used to
    compute Day 2 probabilities even on the Day 4 report).
    """
    records = {team: [0,0,0] for team in sampler.model.ratings.keys()}
    match_predictions = {"a": [], "b": []}
    sq_errs = []
    ref_errs = []
    for group in ["a", "b"]:
        for match_list in matches[group]:
            match_predictions[group].append([])
            for match in match_list:
                match_probs = sampler.get_bo2_probs(match[0], match[1],
                    draw_adjustment=0.05 if not static_ratings else 0.0)
                if match[2] != -1:
                    # match has already been played, so update team
                    # ratings and current records
                    team1 = match[0]
                    team2 = match[1]
                    if not static_ratings:
                        result = (match[2], 2 - match[2])
                        sq_errs.append((1 - match_probs[result[1]])**2)
                        ref_errs.append((1/2)**2 if result[0]==1 else (3/4)**2)
                        sampler.model.update_ratings(team1, team2, result)

                    records[match[0]][2 - match[2]] += 1
                    records[match[1]][match[2]] += 1

                match_predictions[group][-1].append({
                    "teams": (match[0], match[1]), "result": match[2],
                    "probs": [float(f"{p:.4f}") for p in match_probs]})
    if len(sq_errs) > 0:
        brier_skill_score = 1 - ((sum(sq_errs)/len(sq_errs))
                                 / (sum(ref_errs)/len(ref_errs)))
    else:
        brier_skill_score = 0
    return records, match_predictions, brier_skill_score

def predict_matches_dpc(sampler,matches,static_ratings):
    """Code for computing each team's record and the probabilities of
    them wining each match.
    """
    records = {team: [0,0] for team in sampler.model.ratings.keys()}
    match_predictions = {"upper": [], "lower": []}
    sq_errs = []
    for division in ["upper", "lower"]:
        for match_list in matches[division]:
            match_predictions[division].append([])
            for match in match_list:
                team1 = match[0]
                team2 = match[1]
                win_prob = sampler.model.get_win_prob(team1, team2)
                bo3_win_prob = win_prob**2 + (2*win_prob**2*(1 - win_prob))
                if len(match[2]) != 0:
                    # match has already been played, so update team
                    # ratings and current records
                    result = match[2]
                    if isinstance(result[0], str): # default result
                        if result[0] == "W":
                            result = [2, 0]
                        elif result[1] == "W":
                            result = [0, 2]
                        else:
                            result = [0, 0]
                    elif not static_ratings:
                        sampler.model.update_ratings(team1, team2, result)
                        sq_errs.append((int(result[0] > result[1])
                                       - bo3_win_prob)**2)

                    if sum(result) > 0:
                        winner = int(result[1] > result[0])
                        records[match[winner]][0] += 1
                        records[match[1 - winner]][1] += 1

                match_predictions[division][-1].append({
                    "teams": (match[0], match[1]), "result": match[2],
                    "probs": [float(f"{bo3_win_prob:.4f}"),
                              float(f"{1 - bo3_win_prob:.4f}")]})
    if not static_ratings:
        for division in ["upper", "lower"]:
            for match_list in matches["tiebreak"][division].values():
                for team1, team2, result in match_list:
                    sampler.model.update_ratings(team1, team2, result)
    if len(sq_errs) > 0:
        brier_skill_score = 1 - ((sum(sq_errs)/len(sq_errs)) / 0.25)
    else:
        brier_skill_score = 0
    return records, match_predictions, brier_skill_score

def generate_html_dpc(output_file, tabs, title, wildcard_slots):
    """Generates the output forecast report with the provided tabs and
    title. generate_data is used for generating the JSON data files
    used by this report.

    Parameters
    ----------
    output_file : str
        Name of output html file
    tabs : dict
        Dictionary containing the name and suffix of each tab. See
        main() or retroactive_predictions() for an example.
    title : str
        Name to put at the top of the report.
    """
    with open("data/template_dpc.html") as input_f:
        template_str = input_f.read()
    template = Template(template_str, trim_blocks=True, lstrip_blocks=True)

    match_counts = {"upper": [5, 5, 5, 5, 5, 3], "lower": [5, 5, 5, 5, 5, 3]}

    if "weu" in output_file:
        match_counts["lower"][0] = 4
        match_counts["lower"][5] = 4
    if "cn" in output_file:
        match_counts = {"upper": [3, 3, 6, 6, 6, 4],
                        "lower": [7, 7, 4, 4, 4, 2]}
    output = template.render(tabs=tabs, title=title, match_counts=match_counts,
        wildcard_slots=wildcard_slots)
    with open(f"../{output_file}", "w") as output_f:
        output_f.write(output)

def generate_data_dpc(ratings_file, matches, output_file, n_samples, folder, k,
                      wildcard_slots, timestamp="", static_ratings=False):
    """Generates an output JSON file containing the probabilities for
    a given set of matches and input ratings.

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
        List of matches for each division. Each match is a 3-
        element list containing team 1, team 2, and the match
        result as a pair. The pair is empty if the match hasn't
        happened yet. Matches must be provided as a lists of lists,
        where each sub-list contains the matches for a single day.

        Tiebreaker matches should be provide separately as a
        dictionary mapping each boundary tiebreakers were played
        along to the results of those tiebreakers.

        Example:
        {
            "upper": [[
                ["Team A", "Team B", [0, 2]],
                ["Team B", "Team C", [1, 2]]
            ]],
            "lower": [[
                ["Team D", "Team E", [2, 1]],
                ["Team E", "Team D", []]
            ]],
            "tiebreak" :{
                "upper": {
                    "5": [["Team B", "Team C", [2, 0]]]
                }
                "lower": {}
            }
        }
        In this case the results are A 0-2 B, B 1-2 C, D 2-1 E, and
        E vs D has not yet been played.
        An additional match between B and C was played to break the
        tie along spots 5-6 (0-indexed, so this would be between
        ranks 6 and 7). Note that this is in a nested list because
        there may be multiple tiebreak matches (e.g., for a 3-way
        tie).
    output_file : str
        Name of output file to save.
    n_samples : int
        Number of Monte Carlo samples to simulate.
    folder : str
        Folder name to save output to / look for data from.
    k : int
        K parameter for Elo model.
    wildcard_slots : int
        Number of wildcard slots region has.
    timestamp : str, default=""
        Timestamp to put at the top of the report. This is a parameter
        instead of being determined from the current time because I
        generate multiple reports and want them to have the same
        timestamp
    static_ratings : bool, default=False
        If true, ratings will not be updated over the course of a
        simulation. This is used for the fixed-ratings output.
    """
    if output_file == "glicko":
        # static ratings are always used for the Glicko simulator
        # because Glicko explicitly accounts for uncertainty using
        # ratinng deviation.
        sampler = DPCSampler.from_ratings_file_glicko2(ratings_file,
            0.5, static_ratings=True)
    else:
        sampler = DPCSampler.from_ratings_file(ratings_file, k,
            static_ratings=static_ratings)

    with open(f"data/{folder}/teams.json") as team_f:
        teams = json.load(team_f)

    probs = sampler.sample_league(teams, matches, wildcard_slots, n_samples)
    records,match_preds,_ = predict_matches_dpc(sampler,matches,static_ratings)
    ratings = {team: f"{sampler.model.get_team_rating(team):.0f}"
              for team in sampler.model.ratings.keys()}
    output_json = {
        "probs": {
            "group_rank": get_gs_ranks(probs["group_rank"], ["upper","lower"]),
            "record": get_records(probs["record"], probs["point_rank"],
                                  ["upper","lower"]),
            "tiebreak": probs["tiebreak"],
            "matches": match_preds
        },
        "records": records,
        "ratings": ratings,
        "timestamp": timestamp,
        "n_samples": n_samples,
        "model_version": "0.2"
    }
    Path(f"../{folder}/data").mkdir(exist_ok=True)
    with open(f"../{folder}/data/{output_file}.json", "w") as json_f:
        json.dump(output_json, json_f)

def generate_html_ti(output_file, tabs, title):
    """Generates the output forecast report with the provided tabs and
    title. generate_data is used for generating the JSON data files
    used by this report.

    Parameters
    ----------
    output_file : str
        Name of output html file
    tabs : dict
        Dictionary containing the name and suffix of each tab. See
        main() or retroactive_predictions() for an example.
    title : str
        Name to put at the top of the report.
    """
    with open("data/template_ti.html") as input_f:
        template_str = input_f.read()
    template = Template(template_str, trim_blocks=True, lstrip_blocks=True)

    output = template.render(tabs=tabs, title=title,
        match_counts={"a": [12, 8, 12, 4], "b": [8, 12, 8, 8]})
    with open(f"../{output_file}", "w") as output_f:
        output_f.write(output)

def generate_data_ti(ratings_file, matches, output_file, n_samples, folder, k,
                     timestamp="", static_ratings=False, bracket_file=None):
    """Generates an output JSON file containing the probabilities for
    a given set of matches and input ratings.

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
    k : int
        K parameter for Elo model.
    timestamp : str, default=""
        Timestamp to put at the top of the report. This is a parameter
        instead of being determined from the current time because I
        generate multiple reports and want them to have the same
        timestamp
    static_ratings : bool, default=False
        If true, ratings will not be updated over the course of a
        simulation. This is used for the fixed-ratings output.
    bracket_file : str, default=None
        Used for simulating just the elimination bracket. Group stage
        matches must be complete for group stage data to be correct. If
        provided, should be a path to a JSON file containing bracket
        results.
    """
    if output_file == "glicko":
        # static ratings are always used for the Glicko simulator
        # because Glicko explicitly accounts for uncertainty using
        # ratinng deviation.
        sampler = TISampler.from_ratings_file_glicko2(ratings_file,
            0.5, static_ratings=True)
    else:
        sampler = TISampler.from_ratings_file(ratings_file, k,
            static_ratings=static_ratings)

    with open(f"data/{folder}/groups.json") as group_f:
        groups = json.load(group_f)

    if bracket_file is None:
        probs = sampler.sample_group_stage(groups, matches, n_samples)
    else:
        with open(bracket_file) as bracket_f:
            bracket = json.load(bracket_f)
        probs = sampler.sample_main_event(groups, matches,
                                          bracket, n_samples)

    records,match_preds,_ = predict_matches_ti(sampler,matches,static_ratings)
    ratings = {team: f"{sampler.model.get_team_rating(team):.0f}"
              for team in sampler.model.ratings.keys()}
    output_json = {
        "probs": {
            "group_rank": get_gs_ranks(probs["group_rank"], ["a", "b"]),
            "final_rank": get_final_ranks(probs["final_rank"]),
            "record": get_records(probs["record"], probs["point_rank"],
                                  ["a", "b"]),
            "tiebreak": probs["tiebreak"],
            "matches": match_preds
        },
        "records": records,
        "ratings": ratings,
        "timestamp": timestamp,
        "n_samples": n_samples,
        "model_version": "0.2"
    }
    Path(f"../{folder}/data").mkdir(exist_ok=True)
    with open(f"../{folder}/data/{output_file}.json", "w") as json_f:
        json.dump(output_json, json_f)

def generate_html_global_rankings(output_file, dpc_season):
    """Generates a global team rating using the
    /global/elo_ratings.json data file

    Parameters
    ----------
    output_file : str
        Name of output html file
    dpc_season : str
        DPC season used to get team list. This is used to find images,
        so it should be whatever the folder in /data/dpc is called
        (e.g., "sp21")
    """
    with open("data/template_global_rating.html") as input_f:
        template_str = input_f.read()
    template = Template(template_str, trim_blocks=True, lstrip_blocks=True)

    with open("data/global/elo_ratings.json") as json_f:
        team_ratings = json.load(json_f)

    full_name = {
        "na": "North America", "sa": "South America", "weu": "Western Europe",
        "eeu": "Eastern Europe", "cn": "China", "sea": "Southeast Asia"
    }
    team_data = [(team, region, full_name[region], round(rating), last_update)
                 for team, region, rating, last_update in
                 sorted(team_ratings["ratings"], key=lambda x: -x[2])]

    output = template.render(team_data=team_data, dpc_season=dpc_season,
                             timestamp=team_ratings["timestamp"])
    with open(f"../{output_file}", "w") as output_f:
        output_f.write(output)
