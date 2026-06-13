"""Tests for the Phase-B integration-sentiment classifier and auto-score parser."""

from zeta_life.bridge.integration_sentiment import integration_sentiment, parse_felt


class TestSentiment:
    def test_integrated_text_scores_high(self):
        s = integration_sentiment("Me siento coherente, integrada y conectada, con claridad.")
        assert s > 0.7

    def test_fragmented_text_scores_low(self):
        s = integration_sentiment("Todo confuso, disperso y fragmentado, me siento perdida.")
        assert s < 0.3

    def test_neutral_text_is_half(self):
        assert integration_sentiment("Hoy pense en la fotosintesis.") == 0.5

    def test_mixed_text_is_between(self):
        s = integration_sentiment("Algo coherente pero tambien disperso.")
        assert 0.3 < s < 0.7


class TestParseFelt:
    def test_parses_siento_line(self):
        assert parse_felt("...reflexion...\nSIENTO: 0.8") == 0.8

    def test_parses_with_equals_and_case(self):
        assert parse_felt("siento = 0.3") == 0.3

    def test_parses_leading_dot(self):
        assert parse_felt("SIENTO: .5") == 0.5

    def test_clamps_to_unit(self):
        assert parse_felt("SIENTO: 1") == 1.0

    def test_none_when_absent(self):
        assert parse_felt("una reflexion sin auto-score") is None

    def test_parses_embedded_in_text(self):
        assert parse_felt("blah\nAl final SIENTO: 0.62 nada mas") == 0.62
