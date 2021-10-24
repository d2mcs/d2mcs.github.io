"""This module contains implementations for various tiebreaker methods.
"""

import random
from itertools import combinations

class Tiebreaker:
    """Implementations for tiebreaker methods. A simulator is required
    for methods which result in additional matches being played
    (because those matches must be simulated). If additional matches
    will never be required, None can be passed as the simulator object.

    Parameters
    ----------
    simulator : Simulator
        Simulator used for simulating bo1/bo-n matches.
    """
    def __init__(self, simulator):
        self.sim = simulator

    def order_teams(self, teams, point_map):
        """Given a list of teams and a mapping from team names to
        points, creates an ordered list of teams where tied teams
        occupy the same element of the list as a set.

        Parameters
        ----------
        teams : list of str
            list of team names
        point_map : dict
            Mapping between team names and point values

        Returns
        -------
        list
            Ordered list of teams. If two teams have the same number of
            points, those teams are placed in a set at the same
            position of the list. Example (where B and C are tied):

                ["Team A", {"Team B", "Team C"}, "Team D"]
        """
        team_order = []
        for team in sorted(teams, key=lambda t: -point_map[t]):
            if len(team_order) != 0 and isinstance(team_order[-1], set):
                if point_map[next(iter(team_order[-1]))] == point_map[team]:
                    team_order[-1].add(team)
                else:
                    team_order.append(team)
            elif (len(team_order) != 0
                  and point_map[team_order[-1]] == point_map[team]):
                team_order.append({team_order.pop(), team})
            else:
                team_order.append(team)
        return team_order

    def bo1_tiebreak(self, boundary, teams):
        """Breaks a tie along a given boundary using round-robin bo1s.
        Only ties along the boundary are considered, so in the case of
        a multi-way tie (which is when this tiebreak method is used at
        TI) even if teams aren't completely ordered tiebreaks will stop
        once the teams along the boundary are ordered.

        Parameters
        ----------
        boundary : tuple of (int, int)
            Individual boundary to break a tie at. See documentation
            for boundary_tiebreak for more information
        teams : list of str
            list of team names
        """
        tiebreak_points = {team: 0 for team in teams}
        matches = list(combinations(teams, 2))

        random.shuffle(matches)
        for match in matches:
            team1_win = self.sim.sim_bo1(match[0], match[1])
            tiebreak_points[match[1 - int(team1_win)]] += 1
            tiebreak_points[match[int(team1_win)]] -= 1

            # a 3-way tiebreaker may be cut off early if a team goes
            # 2-0 or 0-2 in the first two matches and such a result
            # resolves the tie without needing the third match
            # (see TI8 for an example)
            # in theory something similar could happen with a 4+ way
            # tie but matches might be played simultaneously and there
            # isn't any precedence to determine whether the matches
            # would end early anyway so I explicitly only stop early
            # for 3-way ties
            if len(teams) == 3:
                if (tiebreak_points[match[1-int(team1_win)]] == len(teams) - 1
                      and boundary[0] == 0):
                    break
                if (tiebreak_points[match[int(team1_win)]] == 1 - len(teams)
                      and boundary[1] == len(teams) - 1):
                    break

        team_order = self.order_teams(teams, tiebreak_points)

        rank_start = 0
        for i, team_sets in enumerate(team_order):
            if not isinstance(team_sets, set):
                rank_start += 1
                continue
            rank_end = rank_start + len(team_sets) - 1
            if rank_start <= boundary[0] and rank_end >= boundary[1]:
                break_pos = boundary[0] - rank_start
                relative_boundary = (break_pos, break_pos + 1)
                # if there's still a tie along the boundary,
                # play more bo1s to resolve it
                team_order[i:i+1] = self.bo1_tiebreak(
                    relative_boundary, team_sets)
            rank_start += len(team_sets)

        return team_order

    def boundary_tiebreak(self, tiebreak_boundaries, team_order, point_map):
        """Breaks ties along a boundary by playing additional matches.
        Once the boundary tie has been broken, no additional matches
        are played. Following official TI rules, a bo3 is used for
        2-way ties and bo1s are used for 3+ way ties.

        Ties which aren't along a boundary which requires tiebreak
        matches are not considered (these ties use head-to-head
        results, which are covered by a different function)

        Parameters
        ----------
        tiebreak_boundaries : List of tuple of int
            List of positions (0-indexed) which require additional
            matches in the case of a tie. For example, in the case of
            TI (4th/5th and 8th/9th require tiebreakers) this would be:

                [(3,4), (7,8)]

            The second number is technically redundant but I find it
            more intuitive/easier to remember
        team_order : list
            Team ordering as returned by order_teams
        point_map : dict
            Mapping from team name to current points.

        Returns
        -------
        list
            Updated team ordering with boundary ties resolved
        dict
            dict mapping each boundary to the number of teams tied
            along that boundary
        """
        num_groups = len(team_order) # number of tied groups

        rank_end = len(point_map) - 1
        tiebreak_sizes = {boundary[0]: 0 for boundary in tiebreak_boundaries}
        for i, teams in enumerate(reversed(team_order)):
            # no tie:
            if not isinstance(teams, set):
                rank_end -= 1
                continue

            # determine if tie is along a boundary for which
            # additional matches are played
            boundary_tie = False
            rank_start = rank_end - (len(teams) - 1)
            for boundary in tiebreak_boundaries:
                if rank_start <= boundary[0] and rank_end >= boundary[1]:
                    boundary_tie = True
                    break
            if not boundary_tie:
                rank_end -= len(teams)
                continue

            # there is a tie, and it is along a boundary
            if len(teams) == 2:
                # 2-way tiebreak, play a single bo3
                teams = list(teams)
                result = self.sim.sim_bo_n(3, teams[0], teams[1])
                if result[0] == 2: # team 1 victory
                    team_order[num_groups - i-1:num_groups - i] = [
                        teams[0], teams[1]]
                else:
                    team_order[num_groups - i-1:num_groups - i] = [
                        teams[1], teams[0]]
                rank_end -= 2
                tiebreak_sizes[boundary[0]] = 2
            else:
                # multi-way tiebreak, play bo1s until tiebreak resolved
                break_pos = boundary[0] - rank_start
                relative_boundary = (break_pos, break_pos + 1)
                reordered = self.bo1_tiebreak(relative_boundary, teams)
                team_order[num_groups - i-1:num_groups - i] = reordered

                rank_end -= len(teams)
                tiebreak_sizes[boundary[0]] = len(teams)

        return team_order, tiebreak_sizes

    def h2h_tiebreak(self, h2h_results, team_order, point_map):
        """Breaks any remaining ties in an ordering using head-to-head
        results. Following TI rules, if head-to-head results do not
        resolve the tie results against lower seeded teams are
        considered. If these do not resolve the tie, seeding is
        assigned at random (TI rules specify a coin toss, but it is
        technically possible to have a 3-way unresolved tie).

        Parameters
        ----------
        h2h_results : dict
            dict containing head-to-head results. Example:
            {
                "Team A": {"Team B": 2, "Team C": 1},
                "Team B": {"Team A": 0, "Team C": 2},
                "Team C": {"Team A": 1, "Team B": 0},
            }
            Note that the table should be complete, even though this
            results in redundant information. The memory cost is not
            significant and it makes the code a bit simpler.
        team_order : list
            Team ordering as returned by order_teams
        point_map : dict
            Mapping from team name to current points.

        Returns
        -------
        team_order : list
            Updated team ordering with remaining ties resolved
        """
        # as long as the team order contains a set (unbroken tie),
        # it will have fewer items than the total number of teams
        while len(team_order) != len(point_map):
            # ties must be broken bottom-up because if head-to-head results
            # don't break ties results against lower-seeded teams do, so
            # lower-seeded ties must be broken first
            num_groups = len(team_order)
            for i, teams in enumerate(reversed(team_order)):
                if not isinstance(teams, set):
                    continue
                tiebreak_points = {team: 0 for team in teams}
                # first, check head-to-head
                for (team1, team2) in combinations(teams, 2):
                    tiebreak_points[team1] += h2h_results[team1][team2]
                    tiebreak_points[team2] += h2h_results[team2][team1]

                tie_order = self.order_teams(teams, tiebreak_points)
                if len(tie_order) != 1:
                    team_order[num_groups - i - 1:num_groups - i] = tie_order
                    break # any time a tie is broken, the process restarts

                # head-to-head is even, so check results against lower seeds
                next_seed = len(team_order) - i
                tie_broken = False
                while next_seed < len(team_order):
                    team2 = team_order[next_seed]
                    for team in teams:
                        tiebreak_points[team] += h2h_results[team][team2]
                    tie_order = self.order_teams(teams, tiebreak_points)
                    if len(tie_order) != 1:
                        tie_broken = True
                        break
                    next_seed += 1

                if tie_broken:
                    team_order[num_groups - i - 1:num_groups - i] = tie_order
                    break # any time a tie is broken, the process restarts

                # tie can't be broken, assign at random
                tied_teams = list(teams)
                random.shuffle(tied_teams)
                team_order[num_groups - i - 1:num_groups - i] = tied_teams

        return team_order
