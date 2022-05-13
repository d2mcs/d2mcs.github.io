"""This module contains code for computing various point/rank statistics
related to the DPC season. It is used for the TI Qualification Analysis
post on the website.

Code is provided for reference only. This module does not have a proper
command-line interface.
"""

import json
from bisect import bisect_right
from multiprocessing import Pool

from tqdm import tqdm

from model.sampler import DPCSeasonSampler, DPCMajorSampler

def get_qual_prob(cutoff, rank_point_dist):
    """Computes the marginal probability of a team having a rank <= 12
    (TI qualification) given they have a certain number of points
    """
    prob_cutoff = 0       # P(points=x)
    prob_cutoff_qual = 0  # P(rank <= 12,points=x)

    # Roster changes aren't accounted for by the simulator, so many
    # point cutoffs never appear in the results. To deal with this the
    # nearest lower cutoff is used for probability computation.
    # If the cutoff is in the distribution, closest_below will just be
    # the cutoff itself.
    possible_pts = set()
    for rank in range(len(rank_point_dist)):
        for point, prob in rank_point_dist[rank].items():
            possible_pts.add(point)
    possible_pts = sorted(list(possible_pts))
    closest_below = possible_pts[bisect_right(possible_pts, cutoff) - 1]

    for rank in range(len(rank_point_dist)):
        for point, prob in rank_point_dist[rank].items():
            if point == closest_below:
                prob_cutoff += prob
                if rank <= 11:
                    prob_cutoff_qual += prob
    # P(rank <= 12|points=x) = (rank <= 12,points=x)/P(points=x)
    return prob_cutoff_qual/prob_cutoff

def get_expected_points(rank_point_dist):
    """Computes the sample mean / variance of the point distribution for
    each rank. Specifically:

        mean_r  = sum_x [P(points=x|rank=r)*x]
        var_r   = sum_x [P(points=x|rank=r)*(x - mean_r)^2]

    P(rank=r) = 1/96 for all ranks because there are 96 teams in the
    DPC, so:

        P(points=x|rank=r) = P(points=x,rank=r)/P(rank=r)
                           = P(points=x,rank=r)*96
    """
    sample_dist = []  # mean / variance for each rank
    for rank in range(len(rank_point_dist)):
        sample_dist.append([0,0])
        for points, prob in rank_point_dist[rank].items():
            sample_dist[-1][0] += prob*96*points
    for rank in range(len(rank_point_dist)):
        for points, prob in rank_point_dist[rank].items():
            sample_dist[rank][1] += prob*96*(points - sample_dist[rank][0])**2
    return sample_dist

def get_season_data(season, n_samples, season_format, ratings_type):
    """Collects various statistics using the points / DPC rank
    distribution generated by DPCSeasonSampler.
    """
    teams = {}
    sampler = DPCSeasonSampler.from_ratings_file(
        f"data/dpc/sp{season[:2]}/na/{ratings_type}_ratings.json",
        k=0, static_ratings=True)
    for region in ["na", "sa", "weu", "eeu", "cn", "sea"]:
        with open(f"data/dpc/sp{season[:2]}/{region}/teams.json") as team_f:
            teams[region] = json.load(team_f)
        with open(f"data/dpc/sp{season[:2]}/"
                  f"{region}/{ratings_type}_ratings.json") as rating_f:
            for team, rating in json.load(rating_f).items():
                sampler.model.ratings[team] = rating*0.75

    probs = sampler.sample_season(teams, season_format, n_samples)

    print("Average number of TI-qualified teams with x number of points "
          "earned entirely though majors")
    for points, teams in probs["major_contrib"].items():
        if points > 1000:
            break
        print(f"{points:4d} points: {teams:4.1f} teams")

    print("\nSample mean/std dev of final score values for DPC ranks 1 - 20:")
    expected_points = get_expected_points(probs["rank_point"])
    for i in range(20):
        print(f"{i + 1:2d}: mean = {expected_points[i][0]:.0f}, "
              f"std. dev. = {expected_points[i][1]**(1/2):.0f}")

    print("\nTI qualification probabilities for different point values:")
    for pts in range(700, 1200, 10):
        print(f"{pts:4d}: {get_qual_prob(pts, probs['rank_point'])*100:6.2f}%")

def calculate_major_expected_points():
    """Calculates the number of points a team is expected to earn from a
    major given they are seeded in the group stage / wildcard round /
    playoffs. Assumes a team has a 50% chance of winning each series
    they play.
    """
    expected_points = {"wildcard": [0,0,0], "group stage": [0,0,0],
                       "playoffs": [0,0,0]}
    _, major_allocation = DPCSeasonSampler.get_point_allocation("21-22")

    for tour_idx in range(3):
        points = {
            "1": major_allocation[0][tour_idx],
            "2": major_allocation[1][tour_idx],
            "3": major_allocation[2][tour_idx],
            "4": major_allocation[3][tour_idx],
            "5-6": major_allocation[4][tour_idx],
            "7-8": major_allocation[5][tour_idx]
        }
        finals_val = .5*points["2"] + .5*points["1"]
        lb_f_val = .5*points["3"] + .5*finals_val
        lb_r3_val = .5*points["5-6"] + .5*(.5*points["4"] + .5*lb_f_val)
        lb_r1_val = .5*(.5*points["7-8"] + .5*lb_r3_val)
        ub_val = (.5*(.5*lb_r3_val + .5*(.5*lb_f_val + .5*finals_val))
                  + .5*lb_r1_val)
        # top 2: playoff team, next 4: lower bracket team
        group_stage_val = 2/8*ub_val + 4/8*lb_r1_val
        # top 2: group stage team
        wildcard_val = 2/6*group_stage_val

        expected_points["wildcard"][tour_idx] = wildcard_val
        expected_points["group stage"][tour_idx] = group_stage_val
        expected_points["playoffs"][tour_idx] = ub_val

    for stage, points in expected_points.items():
        print(f"Stage: {stage}, expected points: "
              f"{points[0]:.2f}/{points[1]:.2f}/{points[2]:.2f}")

def simulate_major_expected_points(n_samples):
    """Simulates the major many times and returns the expected point
    value of being seeded at each stage. This can be computed (much
    faster) using direct calculation, but simulation is useful for
    verification.
    """
    sampler = DPCMajorSampler.from_ratings_file(
        "data/dpc/sp21/major/fixed_ratings.json", k=0, static_ratings=True)
    with open("data/dpc/sp21/major/teams.json") as team_f:
        teams = json.load(team_f)

    # get mapping from team name to starting stage
    # (wildcard, group stage, playoffs)
    stage_map = {}
    for stage, team_list in teams.items():
        for team in team_list:
            stage_map[team] = stage

    expected_points = {"wildcard": [0,0,0], "group stage": [0,0,0],
                       "playoffs": [0,0,0]}
    _, major_allocation = DPCSeasonSampler.get_point_allocation("21-22")

    remaining_trials = n_samples
    with tqdm(total=n_samples) as pbar:
        while remaining_trials > 0:
            pool_size = min(1000, remaining_trials)
            remaining_trials -= pool_size

            with Pool() as pool:
                sim_results = [pool.apply_async(sampler.get_sample, (
                        sampler.model, {}, teams, True))
                    for _ in range(pool_size)]
                for sim_result in sim_results:
                    final_ranks, _, _ = sim_result.get()
                    for i, rank in enumerate(["1","2","3","4","5-6","7-8"]):
                        for team in final_ranks[rank]:
                            for tour_idx in range(3):
                                expected_points[stage_map[team]][tour_idx] += (
                                    major_allocation[i][tour_idx]/6/n_samples
                                )
            pbar.update(pool_size)
    for stage, points in expected_points.items():
        print(f"Stage: {stage}, expected points: "
              f"{points[0]:.2f}/{points[1]:.2f}/{points[2]:.2f}")

def main():
    """main function for calling the statistics collection functions"""
    print("ELO PROBABILITIES")
    get_season_data("21", 100000, "21-22", "elo")

    print("\n\nFIXED RATING PROBABILITIES")
    get_season_data("21", 100000, "21-22", "fixed")

    print("\n\nEXPECTED MAJOR POINTS")
    calculate_major_expected_points()
    print("\nsimulating for verification...")
    simulate_major_expected_points(100000)

if __name__ == "__main__":
    main()