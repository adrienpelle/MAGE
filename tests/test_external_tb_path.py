"""Tests for TopAgent's external_tb_path parameter (llm-hw-generator issue #29, ADR 0006
in that project: https://github.com/adrienpelle/llm-hw-generator).

Unlike this repo's other tests/ files (real-run driver scripts), these are genuine pytest
unit tests with every component TopAgent.run_instance() touches - tb_gen, rtl_gen,
sim_reviewer, sim_judge, rtl_edit - replaced with a MagicMock, so no real LLM call or
simulation happens. TopAgent.run_instance() is exercised directly rather than via run()/
_run(), since those only add logging/output-directory setup around it.
"""

from unittest.mock import MagicMock

from mage.agent import TopAgent


def _agent_with_mocked_components(output_dir: str) -> TopAgent:
    agent = TopAgent(llm=MagicMock())
    agent.output_dir_per_run = output_dir
    agent.tb_gen = MagicMock()
    agent.rtl_gen = MagicMock()
    agent.sim_reviewer = MagicMock()
    agent.sim_judge = MagicMock()
    agent.rtl_edit = MagicMock()
    return agent


def test_unset_external_tb_path_calls_tb_gen_as_before(tmp_path):
    agent = _agent_with_mocked_components(str(tmp_path))
    agent.tb_gen.chat.return_value = ("tb code from LLM", "if code from LLM")
    agent.rtl_gen.chat.return_value = (True, "module M; endmodule")
    agent.sim_reviewer.review.return_value = (True, 0, "SIMULATION PASSED")

    is_pass, rtl_code = agent.run_instance(spec="a spec")

    agent.tb_gen.chat.assert_called_once_with("a spec")
    assert (tmp_path / "tb.sv").read_text() == "tb code from LLM"
    assert is_pass is True
    assert rtl_code == "module M; endmodule"


def test_external_tb_path_bypasses_tb_gen_and_uses_supplied_content(tmp_path):
    external_tb = tmp_path / "deterministic_tb.sv"
    external_tb.write_text("module deterministic_tb; endmodule")

    agent = _agent_with_mocked_components(str(tmp_path))
    agent.external_tb_path = str(external_tb)
    agent.rtl_gen.chat.return_value = (True, "module M; endmodule")
    agent.sim_reviewer.review.return_value = (True, 0, "SIMULATION PASSED")

    is_pass, rtl_code = agent.run_instance(spec="a spec")

    agent.tb_gen.chat.assert_not_called()
    assert (tmp_path / "tb.sv").read_text() == "module deterministic_tb; endmodule"
    assert is_pass is True
    assert rtl_code == "module M; endmodule"


def test_external_tb_path_skips_sim_judge_and_tb_gen_on_mismatch(tmp_path):
    """A simulation mismatch against a supplied deterministic TB must never route through
    SimJudge's own TB-regeneration branch, which would otherwise call tb_gen.chat() again
    and silently discard the supplied testbench (see agent.py's comment at this branch).
    """
    external_tb = tmp_path / "deterministic_tb.sv"
    external_tb.write_text("module deterministic_tb; endmodule")

    agent = _agent_with_mocked_components(str(tmp_path))
    agent.external_tb_path = str(external_tb)
    agent.rtl_max_candidates = 1
    agent.rtl_gen.chat.return_value = (True, "candidate rtl")
    # 1st call: the retry loop's own check (fails). 2nd: the one RTL candidate's check
    # (passes). 3rd: MAGE's own `if not is_sim_pass:` re-check after candidate selection
    # (pre-existing MAGE behavior, unrelated to this patch - see agent.py's run_instance).
    agent.sim_reviewer.review.side_effect = [
        (False, 1, "SIMULATION FAILED - 1 MISMATCHES DETECTED"),
        (True, 0, "SIMULATION PASSED"),
        (True, 0, "SIMULATION PASSED"),
    ]

    is_pass, rtl_code = agent.run_instance(spec="a spec")

    agent.sim_judge.chat.assert_not_called()
    agent.tb_gen.chat.assert_not_called()
    assert is_pass is True
    assert rtl_code == "candidate rtl"
