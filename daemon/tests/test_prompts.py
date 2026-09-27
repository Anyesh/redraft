from daemon.prompts import TEMPLATE_VERSION, render
from daemon.sessions import Source

SOURCES = [Source("transcript", "we ship friday"), Source("notes", "priya owns it")]


def test_rederive_holds_sources_and_instruction_but_not_the_draft():
    context, question = render("rederive", "Summarize.", SOURCES, "old draft", [])
    assert "## transcript\nwe ship friday" in context
    assert context.index("transcript") < context.index("notes")
    assert "old draft" not in context
    assert question == "Summarize."


def test_revise_carries_the_edited_draft_and_pinned_lines():
    context, question = render(
        "revise", "Summarize.", SOURCES, "- ship monday", ["- ship monday"]
    )
    assert "## Current draft\n- ship monday" in context
    assert context.rstrip().endswith("- ship monday")
    assert question.startswith("Summarize.")
    assert "edited" in question
    assert "only the complete revised draft" in question
    assert "redundant" in question


def test_pinned_lines_are_listed_in_rederive_too():
    context, _ = render("rederive", "Summarize.", SOURCES, "", ["keep me"])
    assert "keep me" in context


def test_template_version_is_an_int():
    assert isinstance(TEMPLATE_VERSION, int)
