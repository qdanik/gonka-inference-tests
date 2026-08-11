"""The corpus feeds every prompt, so its parsing is worth testing.

Licence boilerplate left in would put the same Gutenberg header at the front of
every document — a shared prefix across all requests, which is exactly what the
prefix cache exploits and what a throughput run must not have.
"""
from __future__ import annotations

from scripts.build_corpus import normalise, strip_boilerplate

HEADER = (
    "The Project Gutenberg eBook of Some Book\n\n"
    "This ebook is for the use of anyone anywhere...\n\n"
    "*** START OF THE PROJECT GUTENBERG EBOOK SOME BOOK ***\n"
)
FOOTER = (
    "\n*** END OF THE PROJECT GUTENBERG EBOOK SOME BOOK ***\n"
    "Updated editions will replace the previous one.\n"
)


class TestStripBoilerplate:
    def test_keeps_only_the_work(self):
        body = strip_boilerplate(HEADER + "Call me Ishmael." + FOOTER)
        assert body == "Call me Ishmael."

    def test_leaves_text_without_markers_alone(self):
        assert strip_boilerplate("  Call me Ishmael.  ") == "Call me Ishmael."

    def test_handles_a_missing_footer(self):
        assert strip_boilerplate(HEADER + "Call me Ishmael.") == "Call me Ishmael."


class TestNormalise:
    def test_joins_hard_wrapped_lines_into_paragraphs(self):
        wrapped = "It is a truth universally\nacknowledged, that a single man\n\nNext one."
        assert normalise(wrapped) == (
            "It is a truth universally acknowledged, that a single man\n\nNext one.")

    def test_drops_blank_paragraphs_and_carriage_returns(self):
        assert normalise("One\r\n\r\n\r\n\r\nTwo") == "One\n\nTwo"

    def test_a_paragraph_break_survives_for_the_reader(self):
        assert normalise("A\nB\n\nC").count("\n\n") == 1
