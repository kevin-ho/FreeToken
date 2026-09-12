from __future__ import annotations

import pytest
import torch

from freetoken.engine.mtp import MTPProposal, run_k1_transaction


class FakeDrafter:
    def __init__(self, token: int, events: list[str]):
        self.token = token
        self.events = events

    def draft_into_batch(self, batch):
        self.events.append("draft")
        logits = torch.full((1, 8), -1.0)
        logits[0, self.token] = 2.0
        return MTPProposal(logits, torch.softmax(logits, dim=-1), 1)


def _logits(token: int) -> torch.Tensor:
    logits = torch.full((1, 8), -1.0)
    logits[0, token] = 2.0
    return logits


def _run(draft: int, target: int, *, abort: bool = False):
    events: list[str] = []
    committed: list[int] = []
    aborted: list[object] = []

    def verify(batch):
        events.append("verify")
        if abort:
            raise RuntimeError("cancelled")
        return _logits(target)

    def commit(batch, result):
        events.append("commit")
        committed.append(result.emitted_token)

    def release(batch):
        events.append("abort")
        aborted.append(batch)

    result = None
    error = None
    try:
        result = run_k1_transaction(
            object(), FakeDrafter(draft, events), verify, commit, release
        )
    except RuntimeError as exc:
        error = exc
    return events, committed, aborted, result, error


def test_k1_draft_precedes_verify_and_emits_one_token_on_accept():
    events, committed, aborted, result, error = _run(3, 3)
    assert events == ["draft", "verify", "commit"]
    assert committed == [3]
    assert not aborted
    assert result.accepted and result.drafted_width == 1
    assert error is None


def test_k1_rejection_commits_target_bonus_token():
    events, committed, aborted, result, error = _run(3, 5)
    assert events == ["draft", "verify", "commit"]
    assert committed == [5]
    assert result is not None and not result.accepted
    assert not aborted
    assert error is None


def test_k1_abort_releases_prepared_batch_once():
    batch = object()
    events: list[str] = []

    def abort(value):
        events.append("abort")
        assert value is batch

    with pytest.raises(RuntimeError, match="cancelled"):
        run_k1_transaction(
            batch,
            FakeDrafter(1, events),
            lambda _: (_ for _ in ()).throw(RuntimeError("cancelled")),
            lambda *_: pytest.fail("commit after abort"),
            abort,
        )
    assert events == ["draft", "abort"]


def test_k1_commit_failure_aborts_prepared_batch():
    batch = object()
    events: list[str] = []

    def abort(value):
        events.append("abort")
        assert value is batch

    def commit(*_):
        events.append("commit")
        raise RuntimeError("commit failed")

    with pytest.raises(RuntimeError, match="commit failed"):
        run_k1_transaction(
            batch, FakeDrafter(1, events), lambda _: _logits(1), commit, abort
        )
    assert events == ["draft", "commit", "abort"]


def test_k1_rejects_wider_drafter_before_verify():
    events: list[str] = []
    released: list[object] = []

    class WideDrafter(FakeDrafter):
        def draft_into_batch(self, batch):
            events.append("draft")
            return MTPProposal(_logits(1), None, 2)

    with pytest.raises(ValueError, match="width 1"):
        run_k1_transaction(
            object(), WideDrafter(1, events), lambda _: pytest.fail("verify"),
            lambda *_: pytest.fail("commit"), released.append,
        )
    assert events == ["draft"]
    assert len(released) == 1
