# d2mcs.github.io

## Overview

This repository contains the source code for d2mcs, a forecasting model for events in the Dota 2 Pro Circuit. Details about the current version of the model can be found on the website's [model information page](https://d2mcs.github.io/model-info/elo.html). The repository also contains the source code of the website itself, including the code used to generate miscellaneous pages such as the [hero popularity tracker](https://d2mcs.github.io/misc/hero_popularity.html).

Note that d2mcs relies on data collected from the Steam WebAPI which I can't publicly share. The components which rely on this data (the Elo model for estimating team ratings, in particular) have been isolated wherever possible to make the rest of the code usable without this dataset. Currently it is possible to generate forecasts using custom ratings for the one tournament that has taken place since d2mcs was released (TI10). Details on how to do so are provided below -- once more information is available about the upcoming DPC season similar instructions will be provided for the Fall leagues.

## Running the TI Simulator

To the best of my knowledge the TI simulator is entirely consistent with official TI rules, including details such as:

- Ties along upper bracket/lower bracket and lower bracket/elimination boundaries are broken with additional matches (bo3 for a 2 way tie, bo1 round-robin for a 3+ way tie).
- Other ties are broken using head to head results, followed by results against lower seeded teams, followed by a coin flip.
- The top seeded teams from the group stage select their opponent from the bottom two seeds in their bracket from the other group (team selection can't really be predicted so the model just decides between the two randomly).
- The elimination bracket correctly orders teams and crosses over for the losers of upper bracket round 2.

Predictions are made both with generic team ratings (i.e., 50% win probability for each match) and ratings based on a modified Elo model (mathematical details below). Unfortunately I can't share match data pulled from the Steam Web API so it is not currently possible to replicate the Elo model's team ratings using the provided code. However, the model's ratings are saved as a plaintext JSON file (in [src/data/ti/10/elo_ratings.json](src/data/ti/10/elo_ratings.json) for TI10) and can be manually modified or replaced with the output of a different model to see how this would affect the output distribution.

Source code is contained in the `src/` folder. Code must be run from that folder. The command line interface currently only generates TI10 predictions (the cli will be updated once DPC forecasts go up), so output is saved to the `ti/10/` folder.

#### Basic Use

For those unfamiliar with git/python a step-by-step guide for Windows is provided [here](https://d2mcs.github.io/guide-windows.pdf)

Python version 3.6+ is required. There are two package dependencies (jinja2 and tqdm) which can be installed using the requirements file (`pip install -r requirements.txt`). Output HTML reports can then be generated with `generate_predictions.py`. Probabilities depend on groups (specified in [src/data/ti/10/groups.json](src/data/ti/10/groups.json), matches (specified in [src/data/ti/10/matches.json](src/data/ti/10/matches.json)), and team ratings (specified in [src/data/ti/10/elo_ratings.json](src/data/ti/10/elo_ratings.json)). The command-line interface requires a single parameter specifying the number of samples to simulate. For example, to run the simulator 10000 times:

```
python generate_predictions.py 10000
```

You can get reasonable estimates with as few as 10,000 samples, but accurate probabilities will generally require more. Python's multiprocessing module is used to take full advantage of all CPU cores. To change the match results listed in matches.json, edit the number that appears after the two team names. 2 indicates a 2-0 result, 1 indicates a 1-1 result, 0 indicates a 0-2 result, and -1 indicates that the match hasn't been played yet (and should be simulated). For example, an unplayed match between Team A and Team B looks like this:

["Team A", "Team B", -1]

If you wanted to change this match to a 2-0 in favor of Team A you would change it to

["Team A", "Team B", 2]

If you want team ratings to be updated based on match results rather than just staying at whatever rating is in the file you can use the `-s` option:

```
python generate_predictions.py -s 10000
```

#### Advanced Use

If you want the tabs linking to other predictions to show up like they do on the website you can generate predictions using the `-f` option

```
python generate_predictions.py -f 10000
```

This will generate predictions with both the elo_ratings.json file and fixed ratings which force every match to use a 50% win probability. If you really want to get into the weeds, you can also adjust how win probabilities are calculated and how rating updates are calculated by editing the `get_win_prob()` and `update_ratings()` methods for the TeamModel object in forecaster.py (lines 443 and 484, respectively -- ignore the versions of these functions in PlayerModel).

**Warning:** the other command line options (e.g., for computing new Elo ratings and generating retroactive results) require the match dataset and cannot be used. You can safely ignore these options.

## Data Attribution
Liquipedia data is used for determining league quality and was also used to collect team and player names for debugging during model development. Match data is collected from the Steam Web API.
