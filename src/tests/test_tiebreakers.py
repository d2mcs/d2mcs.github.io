import pytest
from pathlib import Path

from model.simulator import GroupStage

@pytest.fixture(scope='session')
def sim():
    sim = GroupStage.from_ratings_file(
        "tests/test_ratings.json", k=55, static_ratings=True)
    return sim

class TestH2HTiebreak:
    """Head-to-head tiebreaks. Each test is run several times because
    the tiebreak will revert to a coinflip if it fails, so the tests
    should pass consistently
    """
    def test_basic_h2h(self, sim):
        for _ in range(10):
            h2h = {
                "Team A": {"Team B": 0, "Team C": 2, "Team D": 1},
                "Team B": {"Team A": 2, "Team C": 1, "Team D": 0},
                "Team C": {"Team A": 0, "Team B": 1, "Team D": 0},
                "Team D": {"Team A": 1, "Team B": 2, "Team C": 2},
            }
            team_order = ["Team D", {"Team A", "Team B"}, "Team C"]
            point_map = {"Team A": 3, "Team B": 3, "Team C": 1, "Team D": 5}
            reordered = sim.h2h_tiebreak(h2h, team_order, point_map)
            assert reordered == ["Team D", "Team B", "Team A", "Team C"]

    def test_3x3_h2h(self, sim):
        for _ in range(10):
            h2h = {
                "Team A": {"Team B": 2, "Team C": 2, "Team D": 0, "Team E": 0},
                "Team B": {"Team A": 2, "Team C": 0, "Team D": 2, "Team E": 0},
                "Team C": {"Team A": 0, "Team B": 0, "Team D": 2, "Team E": 2},
                "Team D": {"Team A": 2, "Team B": 0, "Team C": 0, "Team E": 0},
                "Team E": {"Team A": 2, "Team B": 2, "Team C": 0, "Team D": 2},
            }
            team_order = ["Team E", {"Team C", "Team A", "Team B"}, "Team D"]
            point_map = {"Team A": 4, "Team B": 4, "Team C": 4,
                         "Team D": 2, "Team E": 6}
            reordered = sim.h2h_tiebreak(h2h, team_order, point_map)
            assert reordered == ["Team E","Team A","Team B","Team C","Team D"]

    def test_lower_seed_h2h(self, sim):
        for _ in range(10):
            h2h = {
                "Team A": {"Team B": 1, "Team C": 2, "Team D": 1},
                "Team B": {"Team A": 1, "Team C": 1, "Team D": 2},
                "Team C": {"Team A": 0, "Team B": 1, "Team D": 0},
                "Team D": {"Team A": 1, "Team B": 2, "Team C": 0},
            }
            team_order = [{"Team B", "Team A"}, "Team D", "Team C"]
            point_map = {"Team A": 4, "Team B": 4, "Team C": 1, "Team D": 3}
            reordered = sim.h2h_tiebreak(h2h, team_order, point_map)
            # Team B should be higher seed due to having a
            # better result against team D
            assert reordered == ["Team B", "Team A", "Team D", "Team C"]

    def test_random_h2h(self, sim):
        ranks = {
            "Team A": [0, 0, 0], "Team B": [0, 0, 0], "Team C": [0, 0, 0]
        }
        for _ in range(100):
            h2h = {
                "Team A": {"Team B": 1, "Team C": 1, "Team D": 1},
                "Team B": {"Team A": 1, "Team C": 1, "Team D": 1},
                "Team C": {"Team A": 1, "Team B": 1, "Team D": 1}
            }
            team_order = [{"Team B", "Team A", "Team C"}]
            point_map = {"Team A": 3, "Team B": 3, "Team C": 3}
            reordered = sim.h2h_tiebreak(h2h, team_order, point_map)
            # Teams are equal, so tiebreak should be random
            for i in range(3):
                ranks[reordered[i]][i] += 1/100
        for team, probs in ranks.items():
            for prob in probs:
                assert prob >= .1 and prob <= .99

class TestBoundaryTiebreak:
    def test_no_tiebreak_needed(self, sim):
        team_order = ["Team A", "Team B", {"Team C", "Team D"}]
        point_map = {"Team A": 3, "Team B": 2, "Team C": 1, "Team D": 1}

        # tie isn't along the boundary, so nothing should change
        reordered, _ = sim.boundary_tiebreak([(0, 1)], team_order, point_map)
        assert reordered == team_order
        reordered, _ = sim.boundary_tiebreak([(1, 2)], team_order, point_map)
        assert reordered == team_order

    def test_2x2_tiebreak(self, sim):
        probs = {"Team C": 0, "Team D": 0}
        for _ in range(100):
            team_order = ["Team A", "Team B", {"Team C", "Team D"}]
            point_map = {"Team A": 3, "Team B": 2, "Team C": 1, "Team D": 1}
            reordered,_ = sim.boundary_tiebreak([(2,3)], team_order, point_map)
            assert isinstance(reordered[2], str) # ensure tie was broken
            probs[reordered[2]] += 1/100
        # both teams should win ~50 times
        for team, prob in probs.items():
            assert prob >= .1 and prob <= .99

    def test_3x3_tiebreak(self, sim):
        probs = {"Team B": 0, "Team C": 0, "Team D": 0}
        for _ in range(100):
            team_order = ["Team A", {"Team B", "Team C", "Team D"}]
            point_map = {"Team A": 3, "Team B": 1, "Team C": 1, "Team D": 1}
            reordered,_ = sim.boundary_tiebreak([(2,3)], team_order, point_map)
            assert isinstance(reordered[-1], str)
            probs[reordered[-1]] += 1/100

        for team, prob in probs.items():
            assert prob >= .1 and prob <= .99

    def test_4x4_tiebreak(self, sim):
        probs = {"Team A": 0, "Team B": 0, "Team C": 0, "Team D": 0}
        lengths = [0, 0, 0]
        for _ in range(1000):
            team_order = [{"Team A", "Team B", "Team C", "Team D"}]
            point_map = {"Team A": 1, "Team B": 1, "Team C": 1, "Team D": 1}
            reordered,_ = sim.boundary_tiebreak([(2,3)], team_order, point_map)
            assert isinstance(reordered[-1], str) # ensure tie was broken
            lengths[len(reordered) - 3] += 1/1000
            probs[reordered[-1]] += 1/1000

        for team, prob in probs.items():
            assert prob >= .1 and prob <= .99
        # the 4-way tie will sometimes only be partially broken
        for length_prob in lengths:
            assert length_prob >= .01 and length_prob <= .99

    def test_multitie(self, sim):
        upper_probs = {"Team A": 0, "Team B": 0}
        lower_probs = {"Team C": 0, "Team D": 0}
        for _ in range(100):
            team_order = [{"Team A", "Team B"}, {"Team C", "Team D"}]
            point_map = {"Team A": 2, "Team B": 2, "Team C": 1, "Team D": 1}
            reordered,_ = sim.boundary_tiebreak([(0,1), (2,3)],
                team_order, point_map)
            assert len(reordered) == 4 # ensure ties were broken

            upper_probs[reordered[0]] += 1/100
            lower_probs[reordered[2]] += 1/100

        for prob in upper_probs.values():
            assert prob >= .1 and prob <= .99
        for prob in lower_probs.values():
            assert prob >= .1 and prob <= .99
