"""This module contains code for generating an html report using
simulator predictions.
"""

import json

from jinja2 import Template

from model.sampler import TISampler

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

def color_prob(p, color="purple"):
    """Calculates a shade of purple for a probability"""
    if color == "purple":
        hexcode = (hex(int(240 - 120*p))[2:].zfill(2)
              + hex(int(240 - 130*p))[2:].zfill(2)
              + hex(int(240 - 70*p))[2:].zfill(2))
    elif color == "green":
        hexcode = (hex(int(240 - 140*p))[2:].zfill(2)
              + hex(int(240 - 70*p))[2:].zfill(2)
              + hex(int(240 - 140*p))[2:].zfill(2))
    else:
        raise ValueError("Invalid color")
    return hexcode

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
    for team, prob_dict in sorted(rank_probs.items(),
          key=lambda x: sum([i*p for i,p in enumerate(x[1].values())])):
        formatted_probs[team] = []
        for rank, prob in prob_dict.items():
            formatted_probs[team].append({
                    "prob": format_percentage(prob, n_samples),
                    "color": color_prob(prob)})

    return formatted_probs

def format_point_rank_probabilities(point_rank_probs, record_probs, n_samples):
    """formats final point value probabilities"""
    formatted_pr_probs = {"a": {}, "b": {}}
    for group in ["a", "b"]:
        for team, record_map in point_rank_probs[group].items():
            formatted_pr_probs[group][team] = {}
            for record, point_counts in record_map.items():
                formatted_pr_probs[group][team][record] = {}
                record_dict = formatted_pr_probs[group][team][record]
                probs = [format_percentage(p, n_samples)
                         for p in point_counts.values()]
                colors = [color_prob(p) for p in point_counts.values()]
                for i in range(9):
                    record_dict[i+1] = {"prob": probs[i], "color": colors[i]}
    formatted_record_probs = {"a": {}, "b": {}}
    for group in ["a", "b"]:
        for team, records in record_probs[group].items():
            formatted_record_probs[group][team] = []
            for record_prob in records:
                formatted_record_probs[group][team].append({
                        "prob": format_percentage(record_prob, n_samples),
                        "color": color_prob(record_prob, color="green")})
    return formatted_pr_probs, formatted_record_probs

def predict_matches_ti(sampler, matches, static_ratings):
    """Code for computing each team's record and the probabilities of
    them wining each match. Match probabilities are computed only with
    present information (e.g., a team's Day 4 rating is not used to
    compute Day 2 probabilities even on the Day 4 report).
    """
    records = {team: [0,0,0] for team in sampler.model.ratings.keys()}
    formatted_matches = {"a": [], "b": []}
    for group in ["a", "b"]:
        for match_list in matches[group]:
            formatted_matches[group].append([])
            for match in match_list:
                match_probs = sampler.get_bo2_probs(match[0], match[1],
                    draw_adjustment=not static_ratings)
                if match[2] != -1:
                    # match has already been played, so update team
                    # ratings and current records
                    team1 = match[0]
                    team2 = match[1]
                    if not static_ratings:
                        result = (match[2], 2 - match[2])
                        sampler.model.update_ratings(team1, team2, result)

                    records[match[0]][2 - match[2]] += 1
                    records[match[1]][match[2]] += 1

                formatted_probs, colors=format_match_probabilities(match_probs)
                formatted_matches[group][-1].append({
                    "teams": (match[0], match[1]), "result": match[2],
                    "probs": formatted_probs, "colors": colors})
    return records, formatted_matches

def generate_html_ti(ratings_file, matches, output_file, n_samples, folder, k,
                     timestamp="", static_ratings=False, tabs=None,
                     title="The International 10", bracket_file=None):
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
        (group_rank_probs, tiebreak_probs, final_rank_probs, record_probs,
            point_rank_probs) = sampler.sample_group_stage(
                groups, matches, n_samples)
    else:
        with open(bracket_file) as bracket_f:
            bracket = json.load(bracket_f)
        (group_rank_probs, tiebreak_probs, final_rank_probs, record_probs,
            point_rank_probs) = sampler.sample_main_event(
                groups, matches, bracket, n_samples)

    records, formatted_matches = predict_matches_ti(sampler, matches,
                                                    static_ratings)
    ratings = {team: f"{sampler.model.get_team_rating(team):.0f}"
              for team in sampler.model.ratings.keys()}
    formatted_gs_probs = format_gs_rank_probabilities(group_rank_probs,
        n_samples)
    formatted_tie_probs = format_tiebreak_probabilities(
        tiebreak_probs, n_samples)
    formatted_final_probs = format_final_rank_probabilities(
        final_rank_probs, n_samples)
    formatted_point_ranks, record_probs = format_point_rank_probabilities(
        point_rank_probs, record_probs, n_samples)

    with open("data/template.html") as input_f:
        template_str = input_f.read()
    template = Template(template_str)

    output = template.render(gs_prob_strs=formatted_gs_probs,
        final_prob_strs=formatted_final_probs,
        point_ranks_strs=formatted_point_ranks, record_probs=record_probs,
        tiebreak_prob_strs=formatted_tie_probs, records=records,
        ratings=ratings, matches=formatted_matches, timestamp=timestamp,
        n_samples=n_samples, output_file=output_file, tabs=tabs, title=title)

    if tabs is not None:
        with open(f"../{folder}/{output_file}{tabs['active'][1]}",
              "w") as output_f:
            output_f.write(output)
    else:
        with open(f"../{folder}/{output_file}", "w") as output_f:
            output_f.write(output)
