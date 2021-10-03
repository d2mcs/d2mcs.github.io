# d2mcs.github.io

This repository contains code for estimating the probability of various outcomes at The International Dota 2 Championships using Monte Carlo sampling. Current predictions can be found [here](https://d2mcs.github.io/ti10/elo.html). To the best of my knowledge the simulator is entirely consistent with official TI rules, including details such as:

- Ties along upper bracket/lower bracket and lower bracket/elimination boundaries are broken with additional matches (bo3 for a 2 way tie, bo1 round-robin for a 3+ way tie).
- Other ties are broken using head to head results, followed by results against lower seeded teams, followed by a coin flip.
- The top seeded teams from the group stage select their opponent from the bottom two seeds in their bracket from the other group (team selection can't really be predicted so the model just decides between the two randomly).
- The elimination bracket correctly orders teams and crosses over for the losers of upper bracket round 2.

Predictions are made both with generic team ratings (i.e., 50% win probability for each match) and ratings based on a modified Elo model (mathematical details below). Unfortunately I can't share match data pulled from the Steam Web API so it is not currently possible to replicate the Elo model's team ratings using the provided code. However, the model's ratings are saved as a plaintext JSON file (in [data/ti10/elo_ratings.json](data/ti10/elo_ratings.json) for TI10) and can be manually modified or replaced with the output of a different model to see how this would affect the output distribution.

## Running the Code

#### Basic Use

For those unfamiliar with git/python a step-by-step guide for Windows is provided [here](https://d2mcs.github.io/guide-windows.pdf)

Python version 3.6+ is required. There are two package dependencies (jinja2 and tqdm) which can be installed using the requirements file (`pip install -r requirements.txt`). Output HTML reports can then be generated with `generate_predictions.py`. Probabilities depend on groups (specified in [data/ti10/groups.json](data/ti10/groups.json), matches (specified in [data/ti10/matches.json](data/ti10/matches.json)), and team ratings (specified in [data/ti10/elo_ratings.json](data/ti10/elo_ratings.json)). The command-line interface requires a single parameter specifying the number of samples to simulate. For example, to run the simulator 10000 times:

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

This will generate predictions with both the elo_ratings.json file and fixed ratings which force every match to use a 50% win probability. If you really want to get into the weeds, you can also adjust how win probabilities are calculated and how rating updates are calculated by editing the `get_win_prob()` and `update_ratings()` methods for the TeamModel object in forecaster.py (lines 372 and 413, respectively -- ignore the versions of these functions in PlayerModel).

**Warning:** the other command line options (e.g., for computing new Elo ratings and generating retroactive results) require the match dataset and cannot be used. You can safely ignore these options.

## Model
The prediction model uses a standard Elo rating system with the following two modifications:

#### Player-based ratings
Ratings are player-based rather than team-based. Win probabilities and rating adjustments are still calculated using team ratings, where a team's rating is equal to the average of that team's player ratings. This helps deal with some of the issues traditional Elo models run into with Dota 2 (frequent substitutes, organization/team ID changes, roster instability, etc.). The team rating adjustment after each match is distributed unequally among players such that each player's rating drifts towards the team rating -- thus, if a team maintains a consistent roster for a long period of time its players will eventually end up with identical ratings. This is accomplished by calculating each player's share of a rating adjustment as

<img src="https://render.githubusercontent.com/render/math?math=f(p_i) = \frac{r_i}{\sum_jr_j}">&nbsp;&nbsp;if the team loses and
<img src="https://render.githubusercontent.com/render/math?math=f(p_i) = \frac{1/r_i}{\sum_j1/r_j}">&nbsp;&nbsp;if the team wins,

where r<sub>i</sub> is player i's rating and f(p<sub>i</sub>) is player i's share of the team's rating adjustment. Note that player ratings shouldn't be interpreted as individual player skill, because the model can't measure skill gap within a team. The main advantage is that this system is able to handle roster signings and major roster changes much better than a model which only assigns ratings to team IDs. It is also able to make a decent estimate of a newly-formed team's initial rating.

#### K adjustments
Because online matches are far more common than offline matches, the model tends to have a lot of difficulty accounting for regional strength. In general, the more competitive a region is the lower the average rating of its teams will be. This also means that teams which perform well in their region but struggle internationally (e.g. Team Aster) are very difficult to accurately predict. I'm planning on eventually properly accounting for this using region/league quality but for now I'm just using heuristics. The current model uses a dynamic k-factor based on team quality and Liquipedia tier:

- Team quality: similar to chess, the k factor is decreased for higher rated teams. Specifically, the actual k factor scales linearly from 150% of the base k factor for teams with a rating less than or equal to 1000 to 50% of the base k factor for teams with a rating greater than or equal to 2000.
- League quality: because the model was designed to make predictions for high-profile tournaments, it places greater weight on results at high-profile tournaments. Specifically, the k factor decays relative to the tier of the tournament by a factor of t<sup>-.75</sup>, where t is the Liquipedia tier of the tournament.

Intuitively these two adjustments mean that the model is least sensitive to results from high-rated teams in low-quality tournaments and most sensitive to results from low-rated teams in high-quality tournaments.

#### Evaluation

All Elo predictions use a base k parameter of 45 (TI is a tier 1 tournament and a typical TI-quality team has a rating around 1900, so a typical TI match will result in an adjusted k of ~25). I tuned these parameters using all matches up to January 1st, 2017 (the earliest match in my database took place on Feb 1st, 2013), and computed metrics over all tier 1 matches from then until January 1st, 2021. Note that qualifiers are frequently played on the same league ticket as the main event so this includes a lot of open qualifier matches. A plot comparing estimated win probability with actual win probability is shown below:

![Plot comparing estimated win probability to actual win probability for tier 1 matches](image/model_calibration.png)

The vertical lines on the actual probability plot represent 95% standard error bars for the sampled win percentage. The largest error is for the 419 matches which the model predicted the radiant team would win 30% of the time; in actuality, the radiant team won 35.8% of these matches with a standard error of 2.3%. Over all tier 1 matches in the evaluation time frame the model achieved a Brier Skill Score of 0.108. For additional reference, the same plot for all matches in tier 3 or better tournaments is shown below.

![Plot comparing estimated win probability to actual win probability for tier 1 -3 matches](image/model_calibration_tier3.png)

## Data Attribution
Liquipedia data is used for determining league quality and was also used to collect team and player names for debugging during model development. Match data is collected from the Steam Web API.
