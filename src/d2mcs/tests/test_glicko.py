import pytest

from model.forecaster_glicko import Glicko2Model

class TestGlickoExample:
    """A simple sanity check which ensures the Glicko-2 implementation
    computes the same values as the example provided in
    http://glicko.net/glicko/glicko2.pdf
    """
    def test_glicko2_ex(self):
        model = Glicko2Model(0.5)
        model.ratings = {
            "A": ((1500 - 1500)/173.7178, 200/173.7178, 0.06),
            "B": ((1400 - 1500)/173.7178,  30/173.7178, 0.06),
            "C": ((1550 - 1500)/173.7178, 100/173.7178, 0.06),
            "D": ((1700 - 1500)/173.7178, 300/173.7178, 0.06),
        }
        model.update_ratings_batch({
            "A": [("B", 1), ("C", 0), ("D", 0)],
            "B": [("A", 0)],
            "C": [("A", 1)],
            "D": [("A", 1)],
        })
        rating, rd, vol = model.get_team_rating_tuple("A")

        assert rating == pytest.approx(1464.06, 1e-2)
        assert rd == pytest.approx(151.52, 1e-2)
        assert vol == pytest.approx(0.059996, 1e-5)

        assert model.get_team_rating("A") == pytest.approx(1464.06, 1e-2)
