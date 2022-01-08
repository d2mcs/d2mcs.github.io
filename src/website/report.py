"""This module contains code for generating an html report using
simulator predictions.
"""

import json
from pathlib import Path
from datetime import date, datetime

from jinja2 import Template

from model.sampler import (TISampler, DPCLeagueSampler,
                           DPCMajorSampler, DPCSeasonSampler)
from model.forecaster import PlayerModel, TeamModel
from model.forecaster_glicko import Glicko2Model
from model.match_data import MatchDatabase

def format_prob(prob, n_samples):
    """Translates a probability into a neat string"""
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

def color_prob(prob):
    """Generates a purple shade which is darker for higher probs"""
    return (f"rgb({240 - 120*prob:0.0f}, {240 - 130*prob:0.0f}, "
            f"{240 - 70*prob:0.0f})")

def generate_team_ratings_elo(max_tier, k, p, folder, stop_after=None):
    """Code for generating rating estimates for each provided team
    roster. Will only work if matches.db and a list of team rosters
    exist in the data folder.
    """
    match_db = MatchDatabase("data/matches.db")
    player_ids = match_db.get_player_ids()
    id_to_region = match_db.get_id_region_map()
    model = PlayerModel(player_ids, k, p, 0.75, id_to_region,
                        0.1, match_db.predict_match_setting())
    model.compute_ratings(match_db.get_matches(max_tier),stop_after=stop_after)

    regions = {}
    tids = {}
    with open(f"data/{folder}/team_data.json") as tid_f:
        for team, data in json.load(tid_f).items():
            regions[team] = data["region"]
            tids[team] = data["id"]

    with open(f"data/{folder}/rosters.json") as roster_f:
        rosters = json.load(roster_f)

    with open(f"data/{folder}/elo_ratings_lan.json", "w") as lan_f, open(
              f"data/{folder}/elo_ratings_online.json", "w") as online_f:
        for output_f, setting in [(lan_f, "lan"), (online_f, "online")]:
            output_f.write("{\n")
            for i, team in enumerate(rosters.keys()):
                elo_rating = model.get_team_rating(tids[team], rosters[team],
                                                   regions[team], setting)
                if i != len(rosters) - 1:
                    output_f.write(f'  "{team}": {elo_rating:.2f},\n')
                else:
                    output_f.write(f'  "{team}": {elo_rating:.2f}\n')
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
    with open(f"data/{folder}/rosters.json") as roster_f:
        rosters = json.load(roster_f)

    with open(f"data/{folder}/glicko_ratings.json", "w") as output_f:
        output_f.write("{\n")
        for i, (team, tid) in enumerate(team_ids.items()):
            if tid not in model.ratings:
                model.initialize_team(tid, rosters[team])
            mean, rd, sigma = model.get_team_rating_tuple(tid)
            rating = f"[{mean:.2f}, {rd:.4f}, {sigma:.7f}]"
            if i != len(team_ids) - 1:
                output_f.write(f'  "{team}": {rating},\n')
            else:
                output_f.write(f'  "{team}": {rating}\n')
        output_f.write("}\n")

def generate_global_ratings_elo(max_tier, k, p, tour, timestamp):
    """Code for generating an Elo rating for every team which competed
    in the provided DPC season. Ratings will be
    - last recorded rating of the team ID (if the team has played
      under that ID in the last month)
    - current rating of their roster according to the elo_ratings.json
      file for the provided DPC season (remaining teams)
    """
    match_db = MatchDatabase("data/matches.db")
    player_ids = match_db.get_player_ids()
    id_to_region = match_db.get_id_region_map()
    p_model = PlayerModel(player_ids, k, p, tid_region_map=id_to_region)
    rating_history = p_model.compute_ratings(match_db.get_matches(max_tier),
                                             track_history=True)

    team_ids = {}
    regions = {}
    dpc_ratings = {}
    for region in ["na", "sa", "weu", "eeu", "cn", "sea"]:
        folder = f"data/dpc/{tour}/{region}"
        with open(folder + "/team_data.json") as roster_f:
            for team, team_data in json.load(roster_f).items():
                team_ids[team] = team_data["id"]
                regions[team] = region
        with open(folder + "/elo_ratings_lan.json") as lan_rating_f:
            for team, rating in json.load(lan_rating_f).items():
                dpc_ratings[team] = [rating]
        with open(folder + "/elo_ratings_online.json") as online_rating_f:
            for team, rating in json.load(online_rating_f).items():
                dpc_ratings[team].append(rating)

    output_data = {"timestamp": timestamp, "ratings": []}
    current_timestamp = datetime.utcnow().timestamp()
    for team, tid in team_ids.items():
        if tid in rating_history:
            last_rating = rating_history[tid][-1][2]
            if current_timestamp - last_rating < 60*60*24*30:
                output_data["ratings"].append((team, regions[team],
                    float(f"{rating_history[tid][-1][0]:.2f}"),
                    float(f"{rating_history[tid][-1][1]:.2f}"),
                    str(date.fromtimestamp(last_rating))
                ))
            else:
                output_data["ratings"].append((team, regions[team],
                    *dpc_ratings[team], str(date.fromtimestamp(last_rating))
                ))
        else:
            output_data["ratings"].append((team, regions[team],
                                           *dpc_ratings[team], "-"))

    with open("data/global/elo_ratings.json", "w") as output_f:
        json.dump(output_data, output_f)

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

def predict_matches_dpc(sampler, matches, match_file,
                        timestamp, static_ratings):
    """Code for computing each team's record and the probabilities of
    them wining each match.
    """
    try:
        with open(match_file) as match_f:
            existing_preds = json.load(match_f)
    except FileNotFoundError:
        existing_preds = {}

    records = {team: [0,0] for team in sampler.model.ratings.keys()}
    match_predictions = {"upper": [], "lower": []}
    sq_errs = []
    for division in ["upper", "lower"]:
        for match_list in matches[division]:
            match_predictions[division].append([])
            for match in match_list:
                team1 = match[0]
                team2 = match[1]
                if len(match[2]) != 0 and f"{team1}-{team2}" in existing_preds:
                    # if the match has already been played, use the last
                    # recorded win prob
                    win_prob = existing_preds[f"{team1}-{team2}"]["win_prob"]
                else:
                    win_prob = sampler.get_bo_n_win_prob(3, team1, team2)
                    existing_preds[f"{team1}-{team2}"] = {
                        "win_prob": win_prob, "timestamp": timestamp
                    }

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
                    else:
                        sq_errs.append((int(result[0] > result[1])
                                       - win_prob)**2)

                    if sum(result) > 0:
                        winner = int(result[1] > result[0])
                        records[match[winner]][0] += 1
                        records[match[1 - winner]][1] += 1

                match_predictions[division][-1].append({
                    "teams": (team1, team2), "result": match[2],
                    "probs": [float(f"{win_prob:.4f}"),
                              float(f"{1 - win_prob:.4f}")]})
    with open(match_file, "w") as match_f:
        json.dump(existing_preds, match_f)

    if len(sq_errs) > 0:
        brier_skill_score = 1 - ((sum(sq_errs)/len(sq_errs)) / 0.25)
    else:
        brier_skill_score = 0
    return records, match_predictions, brier_skill_score

def predict_matches_major(sampler, matches, static_ratings):
    """Code for computing each team's record and the probabilities of
    them wining each match.
    """
    records = {"wildcard": {}, "group stage": {}}
    match_predictions = {"wildcard": [], "group stage": [], "playoffs": {}}
    sq_errs = []
    ref_errs = []
    for group in ["wildcard", "group stage"]:
        for match_list in matches[group]:
            match_predictions[group].append([])
            for match in match_list:
                for team in match[:2]:
                    if team not in records[group]:
                        records[group][team] = [0,0,0]
                match_probs = sampler.get_bo2_probs(match[0], match[1],
                    draw_adjustment=0.05 if not static_ratings else 0.0)
                if len(match[2]) != 0:
                    result = match[2]
                    if not static_ratings:
                        sq_errs.append((1 - match_probs[result[1]])**2)
                        ref_errs.append((1/2)**2 if result[0]==1 else (3/4)**2)
                        sampler.model.update_ratings(*match)

                    records[group][match[0]][2 - result[0]] += 1
                    records[group][match[1]][2 - result[1]] += 1

                match_predictions[group][-1].append({
                    "teams": (match[0], match[1]), "result": match[2],
                    "probs": [float(f"{p:.4f}") for p in match_probs]})
    for round, match_list in matches["playoffs"].items():
        match_predictions["playoffs"][round] = []
        for match in match_list:
            if round == "GF":
                win_prob = sampler.get_bo_n_win_prob(5, match[0], match[1])
            else:
                win_prob = sampler.get_bo_n_win_prob(3, match[0], match[1])
            if len(match[2]) != 0:
                if not static_ratings:
                    result = match[2]
                    sq_errs.append((int(result[0] > result[1]) - win_prob)**2)
                    ref_errs.append((1/2)**2 if result[0]==1 else (3/4)**2)
                    sampler.model.update_ratings(*match)
            match_predictions["playoffs"][round].append({
                "teams": (match[0], match[1]), "result": match[2],
                "probs": [float(f"{win_prob:.4f}"),
                          float(f"{1 - win_prob:.4f}")]})
    if len(sq_errs) > 0:
        brier_skill_score = 1 - ((sum(sq_errs)/len(sq_errs))
                                 / (sum(ref_errs)/len(ref_errs)))
    else:
        brier_skill_score = 0
    return records, match_predictions, brier_skill_score

def predict_results_season(season, n_samples, ratings_type, k):
    """Generates TI qualification probabilities and estimated DPC points
    over a full DPC season.
    """
    season_format = "21-22"
    if ratings_type == "elo":
        rating_file = f"elo_ratings_lan.json"
    else:
        rating_file = f"fixed_ratings.json"

    teams = {}
    matches = {}
    sampler = DPCSeasonSampler.from_ratings_file(
        f"data/dpc/wn{season}/na/{rating_file}", k=k, static_ratings=False)
    for region in ["na", "sa", "weu", "eeu", "cn", "sea"]:
        with open(f"data/dpc/wn{season}/{region}/teams.json") as team_f:
            teams[region] = json.load(team_f)
        with open(f"data/dpc/wn{season}/{region}/{rating_file}") as rating_f:
            for team, rating in json.load(rating_f).items():
                sampler.model.ratings[team] = rating

    for tour in ["wn", "sp", "sm"]:
        matches[tour] = {}
        for region in ["na", "sa", "weu", "eeu", "cn", "sea", "major"]:
            if tour == "wn":
                match_file = f"data/dpc/{tour}{season}/{region}/matches.json"
            else:
                match_file = (f"data/dpc/{tour}{int(season) + 1}/"
                              f"{region}/matches.json")
            if Path(match_file).exists():
                with open(match_file) as match_f:
                    matches[tour][region] = json.load(match_f)

    probs = sampler.sample_season(teams, season_format, n_samples, matches)
    formatted_probs = []
    for (team, (points, _, qual_prob)) in sorted(probs["final_rank"].items(),
                                                 key=lambda x: -x[1][2]):
        formatted_probs.append([team, format_prob(qual_prob, n_samples),
                                color_prob(qual_prob), f"{points:.0f}"])

    return formatted_probs

def generate_html_dpc(output_file, tabs, title, wildcard_slots, matches):
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
    matches : list
        List of matches. This is used to determine how many match divs
        are needed for each week.
    """
    with open("data/template_dpc.html") as input_f:
        template_str = input_f.read()
    template = Template(template_str, trim_blocks=True, lstrip_blocks=True)

    match_counts = {"upper": [len(mch_list) for mch_list in matches["upper"]],
                    "lower": [len(mch_list) for mch_list in matches["lower"]]}

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
        sampler = DPCLeagueSampler.from_ratings_file_glicko2(ratings_file,
            0.5, static_ratings=True)
    else:
        sampler = DPCLeagueSampler.from_ratings_file(ratings_file, k,
            static_ratings=static_ratings)

    with open(f"data/{folder}/teams.json") as team_f:
        teams = json.load(team_f)

    if len(matches) == 0:
        matches = sampler._random_schedule(teams)

    probs = sampler.sample_league(teams, matches, wildcard_slots, n_samples)
    records,match_preds,_ = predict_matches_dpc(sampler, matches,
        f"data/{folder}/match_predictions_{output_file}.json",
        timestamp, static_ratings)
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
        "model_version": "0.4.1"
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

    with open(f"data/{folder}/teams.json") as group_f:
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
        "model_version": "0.4.1"
    }
    Path(f"../{folder}/data").mkdir(exist_ok=True)
    with open(f"../{folder}/data/{output_file}.json", "w") as json_f:
        json.dump(output_json, json_f)

def generate_html_global_rankings(output_file, tour, n_samples, k):
    """Generates a global team rating using the
    /global/elo_ratings.json data file

    Parameters
    ----------
    output_file : str
        Name of output html file
    tour : str
        DPC tour used to get team list. This is used to find images,
        so it should be whatever the folder in /data/dpc is called
        (e.g., "sp21")
    n_samples : int
        Number of Monte Carlo samples to simulate.
    k : int
        K parameter for Elo model.
    """
    with open("data/template_global_rating.html") as input_f:
        template_str = input_f.read()
    template = Template(template_str, trim_blocks=True, lstrip_blocks=True)

    with open("data/global/elo_ratings.json") as json_f:
        team_ratings = json.load(json_f)

    regions = ["na", "sa", "weu", "eeu", "cn", "sea"]
    region_ratings = {region: {"upper": 0, "lower": 0} for region in regions}
    division_map = {}
    region_map = {}
    for region in regions:
        with open(f"data/dpc/{tour}/{region}/teams.json") as team_f:
            team_data = json.load(team_f)
        for division in ["upper", "lower"]:
            for team in team_data[division]:
                division_map[team] = division
    for team, region, lan_rating, online_rating, _ in team_ratings["ratings"]:
        region_ratings[region][division_map[team]] += lan_rating/8
        region_map[team] = region

    full_name = {
        "na": "North America", "sa": "South America", "weu": "Western Europe",
        "eeu": "Eastern Europe", "cn": "China", "sea": "Southeast Asia"
    }
    team_data = [(team, region, round(lan_rating),
                  round(online_rating), last_update)
                 for team, region, lan_rating, online_rating, last_update in
                 sorted(team_ratings["ratings"], key=lambda x: -x[2])]

    ti_qual_data_elo = predict_results_season(tour[2:], n_samples, "elo", k)
    ti_qual_data_fixed = predict_results_season(tour[2:], n_samples,"fixed", k)
    for i in range(len(ti_qual_data_elo)):
        ti_qual_data_elo[i].append(region_map[ti_qual_data_elo[i][0]])
        ti_qual_data_fixed[i].append(region_map[ti_qual_data_fixed[i][0]])

    output = template.render(team_data=team_data, tour=tour,
                             timestamp=team_ratings["timestamp"],
                             region_ratings=region_ratings,
                             full_name=full_name,
                             ti_qual_data={"elo": ti_qual_data_elo,
                                           "fixed": ti_qual_data_fixed})
    with open(f"../{output_file}", "w") as output_f:
        output_f.write(output)

def generate_data_major(ratings_file, matches, output_file, n_samples, folder,
                        k, timestamp="", static_ratings=False):
    """Generates an output JSON file containing final rank probabilities
    for a major.

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
        List of all matches to be played at the major. The format is
        quite long so instead of documenting it here I refer to the
        file at src/dpc/spring/major/matches.json as an example.

        Note that the dicationary contains 4 keys, the last of which is
        optional (wildcard, group stage, playoffs, tiebreak). Match
        format depends on the stage of the competition.

        If matches for a stage are not provided, the schedule will
        be generated randomly.
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
    """
    if output_file == "glicko":
        sampler = DPCMajorSampler.from_ratings_file_glicko2(ratings_file,
            0.5, static_ratings=True)
    else:
        sampler = DPCMajorSampler.from_ratings_file(ratings_file, k,
            static_ratings=static_ratings)

    with open(f"data/{folder}/teams.json") as team_f:
        teams = json.load(team_f)

    probs = sampler.sample_major(teams, matches, n_samples)

    # By default, pre-tournament group stage probabilities will include
    # teams from the wildcard. These teams should be filtered out until
    # wildcard results are final.
    for team, prob_list in list(probs["group_stage_rank"].items()):
        if sum(prob_list) < (n_samples - 1)/n_samples:
            del probs["group_stage_rank"][team]
    group_rank_probs = get_gs_ranks(
        {"wc": probs["wildcard_rank"], "gs": probs["group_stage_rank"]},
        ["wc", "gs"])
    for _ in range(len(group_rank_probs["gs"]), 8):
        group_rank_probs["gs"].append({"team": "TBD",
                                       "probs": [0 for _ in range(8)]})

    records, _, _ = predict_matches_major(sampler, matches, static_ratings)
    records["group stage"]["TBD"] = [0,0,0]
    for team in teams["wildcard"]:
        if team not in records["wildcard"]:
            records["wildcard"][team] = [0,0,0]
    for team in probs["group_stage_rank"].keys():
        if team not in records["group stage"]:
            records["group stage"][team] = [0,0,0]

    ratings = {team: f"{sampler.model.get_team_rating(team):.0f}"
              for team in sampler.model.ratings.keys()}
    ratings["TBD"] = 0

    output_json = {
        "probs": {
            "group_rank": group_rank_probs,
            "final_rank": get_final_ranks(probs["final_rank"])
        },
        "ratings": ratings,
        "records": records,
        "timestamp": timestamp,
        "n_samples": n_samples,
        "model_version": "0.4.1"
    }
    Path(f"../{folder}/data").mkdir(exist_ok=True)
    with open(f"../{folder}/data/{output_file}.json", "w") as json_f:
        json.dump(output_json, json_f)

def generate_html_major(output_file, tabs, title):
    """Generates the output forecast report with the provided tabs and
    title. generate_data is used for generating the JSON data files
    used by this report.

    Parameters
    ----------
    output_file : str
        Name of output html file
    tabs : dict
        Dictionary containing the name and suffix of each tab.
    title : str
        Name to put at the top of the report.
    """
    with open("data/template_major.html") as input_f:
        template_str = input_f.read()
    template = Template(template_str, trim_blocks=True, lstrip_blocks=True)

    output = template.render(tabs=tabs, title=title)
    with open(f"../{output_file}", "w") as output_f:
        output_f.write(output)
