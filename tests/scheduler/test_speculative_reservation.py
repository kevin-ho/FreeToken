"""Tests for the cache-owned speculative reservation primitive.

These tests cover ownership and request/page-table bookkeeping only. They do not exercise a target
verification forward, model hidden states, SWA shared-tree reconciliation, or end-to-end MTP.
"""
from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Req, SamplingParams
from freetoken.scheduler.cache import CacheManager


def _request(page_table, table_idx=0, prompt_len=4, output_len=8):
    ids = list(range(1, prompt_len + 1))
    cm = CacheManager(page_table.shape[1], 1, page_table, "radix")
    t = torch.tensor(ids, dtype=torch.int32)
    pending = SimpleNamespace(input_ids=t, input_len=len(ids), mm_embeds=None)
    handle = cm.match_req(pending).cuda_handle
    req = Req(input_ids=t, table_idx=table_idx, cached_len=0, output_len=0,
              uid=table_idx, sampling_params=SamplingParams(max_tokens=output_len),
              cache_handle=handle)
    cm.lock(handle)
    req.device_len = prompt_len
    cm.allocate_paged([req])
    req.cached_len = prompt_len
    req.max_device_len = prompt_len + 16
    return cm, req


def test_reservation_allocates_and_abort_restores_state():
    page_table = torch.zeros(1, 16, dtype=torch.int32)
    cm, req = _request(page_table)
    before_free = cm.free_slots.clone()
    before_row = page_table[0].clone()

    reservation = cm.reserve_speculative(req, 4)
    assert req.device_len == 8
    assert len(cm.free_slots) == len(before_free) - 4
    assert torch.equal(page_table[0, 4:8], torch.arange(4, 8))

    reservation.abort()
    assert req.cached_len == 4
    assert req.device_len == 4
    assert torch.equal(page_table[0], before_row)
    assert set(cm.free_slots.tolist()) == set(before_free.tolist())


def test_partial_commit_then_rollback_suffix_keeps_only_committed_pages():
    page_table = torch.zeros(1, 16, dtype=torch.int32)
    cm, req = _request(page_table)
    reservation = cm.reserve_speculative(req, 4)

    reservation.commit_prefix(2)
    assert req.cached_len == 6
    assert req.device_len == 7
    reservation.rollback_suffix()

    assert reservation.reserved_tokens == 2
    assert torch.equal(page_table[0, 4:6], torch.tensor([4, 5], dtype=torch.int32))
    assert torch.equal(page_table[0, 6:8], torch.zeros(2, dtype=torch.int32))
    assert len(cm.free_slots) == 10

    # Aborting after a partial rollback returns even the committed speculative pages.
    reservation.abort()
    assert req.cached_len == 4 and req.device_len == 4
    assert len(cm.free_slots) == 12


def test_double_abort_is_safe():
    page_table = torch.zeros(1, 16, dtype=torch.int32)
    cm, req = _request(page_table)
    before_free = cm.free_slots.clone()
    reservation = cm.reserve_speculative(req, 4)

    reservation.abort()
    reservation.abort()

    assert set(cm.free_slots.tolist()) == set(before_free.tolist())
    assert req.cached_len == 4 and req.device_len == 4


def test_reservation_does_not_evict_prefix_cache_pages():
    page_table = torch.zeros(1, 4, dtype=torch.int32)
    cm, req = _request(page_table, prompt_len=2, output_len=4)
    with pytest.raises(RuntimeError, match="insufficient free pages"):
        cm.reserve_speculative(req, 4)
