from types import SimpleNamespace

import torch

from freetoken.engine.mtp import MTPProposal
from freetoken.scheduler.scheduler import Scheduler


def _logits(token: int) -> torch.Tensor:
    logits = torch.full((1, 8), -1.0)
    logits[0, token] = 2.0
    return logits


def test_flagged_single_request_hook_drafts_before_target_and_commits(monkeypatch):
    events = []
    batch = SimpleNamespace(
        reqs=[object()],
        positions=torch.tensor([0]),
        input_ids=torch.tensor([[4]]),
        hidden_state=torch.ones(1, 2),
        target_lm_head=object(),
    )
    engine = SimpleNamespace(
        prepare_mtp_batch=lambda value, args: True,
        mtp_drafter=SimpleNamespace(
            draft_into_batch=lambda value: (events.append("draft") or MTPProposal(_logits(3), None, 1))
        ),
        verify_mtp_batch=lambda value, args: (events.append("verify") or _logits(3)),
        commit_mtp_batch=lambda value, result: (
            events.append("commit"), setattr(value, "mtp_forward_output", "output")
        ),
        abort_mtp_batch=lambda value: events.append("abort"),
        model=SimpleNamespace(mtp_hidden_state=batch.hidden_state, lm_head=batch.target_lm_head),
    )
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SimpleNamespace(speculative_mtp=True)
    scheduler.engine = engine

    assert scheduler._run_speculative_mtp(batch, object()) == "output"
    assert events == ["draft", "verify", "commit"]


def test_hook_reuses_forward_output_when_hidden_export_is_unavailable():
    calls = []
    batch = SimpleNamespace(reqs=[object()], positions=torch.tensor([0]))
    output = SimpleNamespace(next_tokens_gpu=torch.tensor([5]))
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SimpleNamespace(speculative_mtp=True)
    scheduler.engine = SimpleNamespace(
        prepare_mtp_batch=lambda value, args: (
            calls.append("forward"), setattr(value, "mtp_forward_output", output), False
        )[-1],
        mtp_drafter=SimpleNamespace(draft_into_batch=lambda _: None),
        verify_mtp_batch=lambda *_: calls.append("verify"),
        commit_mtp_batch=lambda *_: calls.append("commit"),
        abort_mtp_batch=lambda *_: calls.append("abort"),
    )

    assert scheduler._run_speculative_mtp(batch, object()) is output
    assert calls == ["forward"]


def test_flagged_hook_fails_closed_without_gemma_inputs():
    batch = SimpleNamespace(reqs=[object()], positions=torch.tensor([0]))
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SimpleNamespace(speculative_mtp=True)
    scheduler.engine = SimpleNamespace(
        prepare_mtp_batch=lambda value, args: False,
        mtp_drafter=SimpleNamespace(draft_into_batch=lambda _: None),
        verify_mtp_batch=lambda *_: None,
        commit_mtp_batch=lambda *_: None,
        model=SimpleNamespace(),
    )

    assert scheduler._run_speculative_mtp(batch, object()) is None
