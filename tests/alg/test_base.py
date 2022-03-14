"""
Test suites for alg.base
"""
import random

from circa.alg.base import Score


class TestScore:
    """
    Test suite for Score
    """

    @staticmethod
    def test_asdict():
        """
        Score.asdict shall provide parameters to create a new Score
        """
        _score = random.random()
        key, value = "key", "test"
        score_key = (1, 0)
        score = Score(_score)
        score[key] = value
        score.key = score_key

        another_score = Score(**score.asdict())
        assert another_score.score == _score
        assert another_score[key] == value
        assert another_score.key == score_key

    @staticmethod
    def test_update():
        """
        Score.update shall update score and info
        """
        _score = 2
        score = Score(_score - 1)
        score["origin"] = "origin value"
        score["common"] = "from score"

        another_score = Score(_score)
        another_score["common"] = "from another"
        another_score["new"] = "new value"

        updated_score = score.update(another_score)
        assert updated_score == score
        assert updated_score.score == _score
        assert updated_score.info == {
            "origin": "origin value",
            "common": "from another",
            "new": "new value",
        }
