"""Tests for RTLEditor's JSON decode guard (llm-hw-generator issue #62:
https://github.com/adrienpelle/llm-hw-generator).

`parse_output` called `json.loads` on the raw model response with no guard, so an empty or
malformed response raised an uncaught `JSONDecodeError` out of `chat()` and took down the whole
TopAgent pipeline. Both siblings already handled this - `RTLGenerator.parse_output` catches and
lets its caller drop the candidate, `TBGenerator` retries `json_decode_max_trial` times - and
the editor was the only one of the three that did not.

Genuine pytest unit tests with no real LLM call or simulation, following
`tests/test_external_tb_path.py`'s precedent in this fork rather than the real-run driver
scripts the rest of `tests/` contains.
"""

from unittest.mock import MagicMock

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole

from mage.rtl_editor import JSON_DECODE_ERROR_PREFIX, RTLEditor

_VALID_STEP = (
    '{"reasoning": "looks fine", '
    '"action_input": {"command": "replace_content", "args": {"a": 1}}}'
)


def _response(content: str) -> ChatResponse:
    return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=content))


def _editor() -> RTLEditor:
    return RTLEditor(token_counter=MagicMock(), sim_reviewer=MagicMock())


# --------------------------------------------------------------------------- #
# parse_output
# --------------------------------------------------------------------------- #


def test_parse_output_reports_an_empty_response_instead_of_raising():
    # `json.loads("")` - the exact failure seen in llm-hw-generator issue #57's real run,
    # `Expecting value: line 1 column 1 (char 0)`, which crashed all three outer attempts.
    output = _editor().parse_output(_response(""))

    assert output.reasoning.startswith(JSON_DECODE_ERROR_PREFIX)
    assert "Expecting value" in output.reasoning


def test_parse_output_reports_a_malformed_but_non_empty_response():
    output = _editor().parse_output(_response('{"reasoning": "half a response", '))

    assert output.reasoning.startswith(JSON_DECODE_ERROR_PREFIX)


def test_parse_output_still_parses_a_valid_response_unchanged():
    output = _editor().parse_output(_response(_VALID_STEP))

    assert output.reasoning == "looks fine"
    assert output.action_input.command == "replace_content"
    assert output.action_input.args == {"a": 1}


# --------------------------------------------------------------------------- #
# chat(): retry, then give up as a reported failure
# --------------------------------------------------------------------------- #


def _editor_for_chat(tmp_path, responses: list[str]) -> RTLEditor:
    """An editor whose `generate` replays `responses` in order, with the filesystem and
    action-running side of `chat()` stubbed out - this is about response handling only.
    """
    editor = _editor()
    (tmp_path / "rtl.sv").write_text("module M; endmodule")
    (tmp_path / "tb.sv").write_text("tb")
    editor.generate = MagicMock(side_effect=[_response(text) for text in responses])
    editor.get_init_prompt_messages = MagicMock(return_value=[])
    editor.get_order_prompt_messages = MagicMock(return_value=[])
    editor.run_action = MagicMock(return_value={"is_action_executed": True})
    editor.get_action_output_message = MagicMock(
        return_value=[ChatMessage(role=MessageRole.USER, content="x")] * 2
    )
    return editor


def _chat(editor: RTLEditor, tmp_path):
    return editor.chat(
        spec="a spec",
        output_dir_per_run=str(tmp_path),
        sim_failed_log="SIMULATION FAILED - 1 MISMATCHES DETECTED",
        sim_mismatch_cnt=1,
    )


def test_chat_retries_after_an_undecodable_response_and_then_proceeds(tmp_path):
    editor = _editor_for_chat(tmp_path, ["", _VALID_STEP])
    editor.is_done = True

    is_pass, rtl_code = _chat(editor, tmp_path)

    # Two model turns: the undecodable one, then the good one it retried into. The action is
    # only ever run on the decodable step - never on garbage.
    assert editor.generate.call_count == 2
    editor.run_action.assert_called_once()
    assert is_pass is True
    assert rtl_code == "module M; endmodule"


def test_chat_gives_up_as_a_reported_failure_rather_than_raising(tmp_path):
    # Persistently undecodable. Before the guard this raised JSONDecodeError out of chat();
    # now it ends the repair loop and hands back the RTL that is actually on disk, which is
    # the difference between "this module failed" and "the pipeline crashed".
    editor = _editor_for_chat(tmp_path, [""] * 5)

    is_pass, rtl_code = _chat(editor, tmp_path)

    assert is_pass is False
    assert rtl_code == "module M; endmodule"
    assert editor.generate.call_count == editor.json_decode_max_trial
    editor.run_action.assert_not_called()


def test_chat_does_not_burn_the_whole_trial_budget_on_decode_failures(tmp_path):
    # The decode budget is separate from and much smaller than max_trials, so an unusable
    # model cannot spend 15 rounds saying nothing.
    editor = _editor_for_chat(tmp_path, [""] * 20)

    _chat(editor, tmp_path)

    assert editor.json_decode_max_trial < editor.max_trials
    assert editor.generate.call_count == editor.json_decode_max_trial
