"""Tests for TopAgent's external_tb_prompt_path (llm-hw-generator issue #63:
https://github.com/adrienpelle/llm-hw-generator).

The prompt carries the testbench the model must satisfy, and an exhaustively-enumerated one
can exceed the model's whole input budget by itself - StateBeacon's is 2.2MB against a
250k-token-per-minute limit, and its top module could not be generated at all. This lets the
caller supply an abbreviated copy for the prompt while simulation keeps using the full one.

The distinction these tests exist to pin: what the model is SHOWN may shrink; what the RTL is
CHECKED against must not. Getting that backwards would silently weaken every verification.

Genuine pytest unit tests with every component mocked, following test_external_tb_path.py.
"""

from unittest.mock import MagicMock

from mage.agent import TopAgent

_FULL_TB = "// full testbench\n" + "\n".join(f"// case {i}" for i in range(100))
_SHOWN_TB = "// abbreviated testbench\n// case 0\n// ... 99 further cases omitted"


def _agent_with_mocked_components(output_dir: str) -> TopAgent:
    agent = TopAgent(llm=MagicMock())
    agent.output_dir_per_run = output_dir
    agent.tb_gen = MagicMock()
    agent.rtl_gen = MagicMock()
    agent.sim_reviewer = MagicMock()
    agent.sim_judge = MagicMock()
    agent.rtl_edit = MagicMock()
    agent.rtl_gen.chat.return_value = (True, "module M; endmodule")
    agent.sim_reviewer.review.return_value = (True, 0, "SIMULATION PASSED")
    return agent


def _write(tmp_path, name: str, text: str) -> str:
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_prompt_gets_the_abbreviated_tb_while_simulation_gets_the_full_one(tmp_path):
    agent = _agent_with_mocked_components(str(tmp_path))
    agent.external_tb_path = _write(tmp_path, "full.sv", _FULL_TB)
    agent.external_tb_prompt_path = _write(tmp_path, "shown.sv", _SHOWN_TB)

    agent.run_instance(spec="a spec")

    # The model saw the short one...
    assert agent.rtl_gen.chat.call_args.kwargs["testbench"] == _SHOWN_TB
    # ...and the file every simulation runs against is still the complete one.
    assert (tmp_path / "tb.sv").read_text() == _FULL_TB


def test_without_a_prompt_tb_the_model_sees_the_full_testbench_as_before(tmp_path):
    agent = _agent_with_mocked_components(str(tmp_path))
    agent.external_tb_path = _write(tmp_path, "full.sv", _FULL_TB)
    agent.external_tb_prompt_path = None

    agent.run_instance(spec="a spec")

    assert agent.rtl_gen.chat.call_args.kwargs["testbench"] == _FULL_TB
    assert (tmp_path / "tb.sv").read_text() == _FULL_TB


def test_every_rtl_generation_call_gets_the_abbreviated_tb_including_candidates(tmp_path):
    """The candidate-regeneration branch fires whenever a first draft fails simulation, and
    it makes `rtl_max_candidates` further requests. Leaving those on the full testbench would
    reproduce the exact budget overrun this exists to remove - and would do it *before*
    RTLEditor is ever reached, so no amount of capping the repair loop would help.

    The first version of this change missed both of those call sites; the test that missed
    them asserted only the first draft, because `sim_reviewer.review()` returned a pass and
    the branch was never entered.
    """
    agent = _agent_with_mocked_components(str(tmp_path))
    agent.external_tb_path = _write(tmp_path, "full.sv", _FULL_TB)
    agent.external_tb_prompt_path = _write(tmp_path, "shown.sv", _SHOWN_TB)
    agent.rtl_max_candidates = 3
    # Fail the first simulation so candidate regeneration is entered, then pass.
    agent.sim_reviewer.review.side_effect = [
        (False, 2, "SIMULATION FAILED - 2 MISMATCHES DETECTED"),
        (True, 0, "SIMULATION PASSED"),
    ] + [(True, 0, "SIMULATION PASSED")] * 8
    agent.sim_judge.chat.return_value = MagicMock(is_rtl_bug=True)
    agent.rtl_gen.gen_candidates.return_value = [(True, "module M; endmodule")] * 2

    agent.run_instance(spec="a spec")

    shown_everywhere = [
        call.kwargs["testbench"] for call in agent.rtl_gen.chat.call_args_list
    ] + [call.kwargs["testbench"] for call in agent.rtl_gen.gen_candidates.call_args_list]
    assert shown_everywhere, "expected at least one generation call"
    assert all(tb == _SHOWN_TB for tb in shown_everywhere), shown_everywhere
    # ...and the simulated file is still complete.
    assert (tmp_path / "tb.sv").read_text() == _FULL_TB


def test_the_repair_loop_actually_reads_the_abbreviated_tb(tmp_path):
    """Asserts the substitution, not just that the path was assigned - the earlier version of
    this test stopped at the attribute, one step short of the behaviour it claimed to guard.
    """
    from mage.rtl_editor import RTLEditor

    editor = RTLEditor(
        token_counter=MagicMock(),
        sim_reviewer=MagicMock(),
        prompt_tb_path=_write(tmp_path, "shown.sv", _SHOWN_TB),
    )
    editor.spec = "a spec"
    editor.sim_failed_log = "SIMULATION FAILED - 1 MISMATCHES DETECTED"
    editor.tb_path = _write(tmp_path, "full.sv", _FULL_TB)

    prompt = "".join(m.content for m in editor.get_init_prompt_messages())

    assert _SHOWN_TB in prompt
    assert _FULL_TB not in prompt


def test_the_repair_loop_falls_back_to_tb_sv_when_no_abbreviated_copy_is_given(tmp_path):
    from mage.rtl_editor import RTLEditor

    editor = RTLEditor(token_counter=MagicMock(), sim_reviewer=MagicMock())
    editor.spec = "a spec"
    editor.sim_failed_log = "failed"
    editor.tb_path = _write(tmp_path, "full.sv", _FULL_TB)

    prompt = "".join(m.content for m in editor.get_init_prompt_messages())

    assert _FULL_TB in prompt


def test_run_threads_the_prompt_tb_path_to_the_repair_loop_too(tmp_path):
    # rtl_editor re-reads tb.sv into its own prompt, so capping only the generator would
    # leave every repair round still over budget - and for an exhaustive testbench the repair
    # loop is exactly where the long tail of rounds happens.
    agent = TopAgent(llm=MagicMock())
    agent.token_counter = MagicMock()
    agent.run_instance = MagicMock(return_value=(True, "module M; endmodule"))
    agent.set_output_path(str(tmp_path / "out"))
    agent.set_log_path(str(tmp_path / "log"))

    agent.run(
        benchmark_type_name="t",
        task_id="m",
        spec="a spec",
        external_tb_path=_write(tmp_path, "full.sv", _FULL_TB),
        external_tb_prompt_path=_write(tmp_path, "shown.sv", _SHOWN_TB),
    )

    assert agent.rtl_edit.prompt_tb_path == str(tmp_path / "shown.sv")
