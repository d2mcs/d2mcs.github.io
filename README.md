# d2mcs.github.io

## Overview

This repository contains the source code for d2mcs, a forecasting model for events in the Dota 2 Pro Circuit. Details about the current version of the model can be found on the website's [model information page](https://d2mcs.github.io/model-info/elo.html). The repository also contains the source code of the website itself, including the code used to generate miscellaneous pages such as the [hero popularity tracker](https://d2mcs.github.io/misc/hero_popularity.html).

Note that d2mcs relies on data collected from the Steam WebAPI which I can't publicly share. The components which rely on this data (the Elo model for estimating team ratings, in particular) have been isolated wherever possible to make the rest of the code usable without this dataset. Instructions on how to generate custom forecasts with your own team ratings are provided below.

## Generating a Custom Forecast

#### Basic Use

All source code is contained in the `src/` folder. The following instructions assume you are working from this folder, though the main script can be run from anywhere.

Python version 3.6+ is required. There are two package dependencies (jinja2 and tqdm) which can be installed using the requirements file (`pip install -r requirements.txt`). Output HTML reports can then be generated with `generate_predictions.py`. The basic syntax is `generate_predictions.py EVENT SAMPLE_COUNT`, where event is the event you want to generate predictions for and sample count is the number of Monte Carlo samples you want to simulate. For example, to generate predictions for TI10 with 10,000 samples:

```
python generate_predictions.py ti10 10000
```

If you're generating predictions for a DPC league, you must also specify the region using `-r`:

```
python generate_predictions.py dpc-sp21 10000 -r na
```

#### Adjusting Team Ratings and Match Results

Each event has three key files of interest: `ratings.json`, `matches.json`, and `teams.json`. All three files are contained in the data folder of each event. For TI 10, for example, this folder is `src/data/ti/10`. For the NA Spring DPC league, it would be `src/data/sp21/na`.

`teams.json` describes how teams are divided into group A or B (for TI) or upper/lower divisions (for DPC leagues). This file generally should not be changed (unless you want to are generating a forecast for a hypothetical event with different groups/matches). `ratings.json` contains the rating for each team, and can be modified to make teams more or less likely to win matches in the simulations. For reference, here is how the rating difference between two teams maps to win probability:

| Rating Advantage  | Win Probability |
| ----------------- | --------------- |
| 500               | 94.7%           |
| 400               | 90.9%           |
| 300               | 84.9%           |
| 200               | 76.0%           |
| 100               | 64.0%           |
| 50                | 57.1%           |
| 0                 | 50.0%           |

`matches.json` contains a list of matches and, once those matches have been played, their results. For TI 7-10 group stages, results must be one of 0, 1, 2, or -1, where 2 indicates a 2-0 result, 1 indicates a 1-1 result, 0 indicates a 0-2 result, and -1 indicates that the match hasn't been played yet. For example, an unplayed match between Team A and Team B looks like this:

`["Team A", "Team B", -1]`

If you wanted to change this match to a 2-0 in favor of Team A you would change it to

`["Team A", "Team B", 2]`

This format is no longer being used for new tournaments. For all other match results, results should be a pair containing the score of each team. For example, a 2-0 results would look like this:

`["Team A", "Team B", [2, 0]]`

Unplayed matches are simply empty lists:

`["Team A", "Team B", []]`

By default, whatever ratings you are using in the `ratings.json` file will be assumed to be "correct," and will not be changed by match results. If you want team ratings to be updated based on match results rather you can use the `-s` option:

```
python generate_predictions.py -s 10000
```

This will also update team ratings based on simulated results during the simulation, which will widen the distribution of possible results somewhat.

## Data Attribution
Liquipedia data is used for determining league quality and was also used to collect team and player names for debugging during model development. Match data is collected from the Steam Web API.
