"""This module contains the command-line interface for generating
hero popularity reports.
"""

import sqlite3
from datetime import datetime, date
import json
import argparse

from jinja2 import Template
import plotly
import plotly.graph_objects as go

def generate_draft_data(match_database, max_tier=3, bans=None):
    """Generates hero popularity data from the match database. This is
    just so I can provide the data publically.
    """
    con = sqlite3.connect(match_database)
    cur = con.cursor()

    hero_listing = cur.execute("SELECT hero_id, name FROM heroes")
    hero_map = {}
    for hero_id, name in hero_listing:
        hero_map[hero_id] = name

    draft_data = cur.execute("""SELECT
        radiant_hero1, radiant_hero2, radiant_hero3, radiant_hero4,
        radiant_hero5, dire_hero1, dire_hero2, dire_hero3, dire_hero4,
        dire_hero5,
        radiant_ban1, dire_ban1, radiant_ban2, dire_ban2,
        radiant_ban3, dire_ban3, radiant_ban4, dire_ban4,
        radiant_ban5, dire_ban5, radiant_ban6, dire_ban6,
        radiant_ban7, dire_ban7,
        timestamp, liquipediatier.tier
        FROM matches JOIN liquipediatier
             ON matches.league_id = liquipediatier.league_id
        ORDER BY timestamp""")
    dates = []
    popularities = []
    running_pop = {}
    match_counter = 0
    for row in draft_data:
        timestamp = row[-2]
        if row[10] == 0:
            # first ban is empty, so ignore this match (probably a
            # remake or abandoned match)
            continue
        if row[-1] > max_tier:
            continue
        if bans is None:
            hero_list = row[:10]
        elif bans == "first-phase":
            if timestamp < datetime.fromisoformat("2017-10-31").timestamp():
                hero_list = row[:14] # switch to 3 first-phase bans in 7.07
            elif timestamp < datetime.fromisoformat("2020-03-17").timestamp():
                hero_list = row[:16] # switch to 4 first-phase bans in 7.25
            elif timestamp < datetime.fromisoformat("2020-06-28").timestamp():
                hero_list = row[:18] # back to 2 first-phase bans in 7.27
            else:
                hero_list = row[:14]
        elif bans == "all":
            hero_list = row[:-2]
        else:
            raise ValueError("Invalid ban configuration string")
        for hero_id in hero_list:
            if hero_id != 0:
                hero = hero_map[hero_id]
                running_pop[hero] = running_pop.get(hero, 0) + 1

        match_counter = (match_counter + 1) % 200
        if match_counter == 0:
            timestamp = row[-2]
            dates.append(timestamp)
            current_popularity = {}
            for hero in hero_map.values():
                current_popularity[hero] = running_pop.get(hero, 0)/200
            popularities.append({
                "timestamp": timestamp, "pick_rate": current_popularity
            })
            running_pop = {}
    if bans is not None:
        output_path = f"data/hero-popularity/popularity-{bans}.json"
    else:
        output_path = "data/hero-popularity/popularity.json"
    with open(output_path, "w") as out_f:
        json.dump(popularities, out_f)

def update_match_data():
    """Updates the publically available pick rate JSON files. This will
    only work if the (non-public) match database is in the data folder.
    """
    generate_draft_data("data/matches.db")
    generate_draft_data("data/matches.db", bans="first-phase")
    generate_draft_data("data/matches.db", bans="all")
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    with open("data/hero-popularity/metadata.json", "w") as out_f:
        out_f.write(f'{{"last-update": "{timestamp}"}}\n')

def make_hero_popularity_page():
    """Generates plots of the pick rate data using plotly and renders
    the HTML page containing them.
    """
    plot_divs = {}
    for data_file in ["popularity","popularity-first-phase","popularity-all"]:
        with open(f"data/hero-popularity/{data_file}.json") as data_f:
            hero_data = json.load(data_f)
        fig = go.Figure()
        dates = [date.fromtimestamp(row["timestamp"]) for row in hero_data]
        for hero in sorted(hero_data[0]["pick_rate"].keys()):
            hero_freq = [row["pick_rate"][hero] for row in hero_data]

            if hero == "Abaddon":
                fig.add_trace(go.Scatter(x=dates, y=hero_freq, name=hero,
                                         fill='tozeroy'))
            else:
                fig.add_trace(go.Scatter(x=dates, y=hero_freq, name=hero,
                                         fill='tozeroy', visible="legendonly"))
        fig.update_layout(xaxis_title="Date",
                          yaxis_title="Percent of Matches Picked/Banned")
        plot_divs[data_file] = plotly.io.to_html(fig,
            include_plotlyjs="cdn" if data_file == "popularity" else False,
            full_html=False, config={"responsive": True})

    with open(f"data/hero-popularity/metadata.json") as metadata_f:
        timestamp = json.load(metadata_f)["last-update"]

    with open("data/template_hero_popularity.html") as input_f:
        template_str = input_f.read()
    template = Template(template_str, trim_blocks=True, lstrip_blocks=True)
    output = template.render(plot_divs=plot_divs, timestamp=timestamp)
    with open(f"../misc/hero_popularity.html", "w") as output_f:
        output_f.write(output)

def main():
    parser = argparse.ArgumentParser(
        description="Generate hero popularity report for pro Dota 2 matches.")
    parser.add_argument("-r","--regen-data", action='store_true',
        default=False, help="Regenerates JSON data files. This will only work "
        "if matches.db exists in the data folder.")
    args = parser.parse_args()

    if args.regen_data:
        update_match_data()
    make_hero_popularity_page()

if __name__ == "__main__":
    main()
