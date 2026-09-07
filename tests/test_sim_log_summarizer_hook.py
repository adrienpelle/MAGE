"""Tests for TopAgent/RTLEditor's sim_log_summarizer hook (llm-hw-generator issue #64).

A simulation log scales with the stimulus count and MAGE places it in four prompts. This fork
takes a *hook* rather than owning a reduction policy: the caller emits the log format, so the
caller decides what is noise. Left unset, nothing is reduced and every class behaves exactly
as it did, which is the contract ADR 0006 states for this fork.

What the tests pin is that the hook reaches every prompt path - the earlier version reached
only the editor's one-shot init prompt and missed its per-round action output, which is
carried every round and is by far the larger half of the cost.
"""

import json
from unittest.mock import MagicMock

from mage.agent import TopAgent
from mage.rtl_editor import RTLEditor

_RAW = json.dumps({"stdout": "Match: a=1\nMISMATCH: b=2\nSIMULATION FAILED - 1 MISMATCHES DETECTED", "stderr": ""})
_SHOWN = "REDUCED"


def _editor(summarizer=None) -> RTLEditor:
    return RTLEditor(
        token_counter=MagicMock(), sim_reviewer=MagicMock(), sim_log_summarizer=summarizer
    )


def test_the_init_prompt_shows_the_reduced_log(tmp_path):
    editor = _editor(lambda log: _SHOWN)
    editor.spec = "a spec"
    editor.sim_failed_log = _RAW
    tb = tmp_path / "tb.sv"
    tb.write_text("// tb")
    editor.tb_path = str(tb)

    prompt = "".join(m.content for m in editor.get_init_prompt_messages())

    assert _SHOWN in prompt
    assert "Match: a=1" not in prompt
    assert editor.sim_failed_log == _RAW  # the decision-making copy is untouched


def _editor_past_the_syntax_check(tmp_path, summarizer=None) -> RTLEditor:
    """`replace_sanity_check` returns early on a syntax failure, so the simulation branch -
    the one that carries the log - is only reachable with RTL that actually compiles."""
    editor = _editor(summarizer)
    rtl = tmp_path / "rtl.sv"
    rtl.write_text("module M(input logic a, output logic b);\n  assign b = a;\nendmodule\n")
    editor.rtl_path = str(rtl)
    editor.sim_reviewer.dependency_rtl_paths = None
    editor.sim_reviewer.review.return_value = (False, 1, _RAW)
    return editor


def test_the_per_round_action_output_shows_the_reduced_log(tmp_path):
    """The one the first attempt missed. This dict is json.dumps'd into a USER message every
    editing round, so at 15 rounds it dwarfs the single init prompt."""
    editor = _editor_past_the_syntax_check(tmp_path, lambda log: _SHOWN)

    output = editor.replace_sanity_check()

    assert output["is_syntax_pass"] is True, "test needs to reach the simulation branch"
    assert output["error_msg"] == _SHOWN


def test_without_a_summarizer_every_path_is_byte_for_byte_unchanged(tmp_path):
    # ADR 0006's contract for this fork: unset means behaves exactly as before.
    editor = _editor_past_the_syntax_check(tmp_path)
    editor.spec = "a spec"
    editor.sim_failed_log = _RAW
    tb = tmp_path / "tb.sv"
    tb.write_text("// tb")
    editor.tb_path = str(tb)

    assert _RAW in "".join(m.content for m in editor.get_init_prompt_messages())
    assert editor.replace_sanity_check()["error_msg"] == _RAW


def test_the_agent_reduces_the_judge_and_failed_trial_logs_too():
    agent = TopAgent(llm=MagicMock())
    agent.sim_log_summarizer = lambda log: _SHOWN

    assert agent._shown_sim_log(_RAW) == _SHOWN
    assert TopAgent(llm=MagicMock())._shown_sim_log(_RAW) == _RAW
