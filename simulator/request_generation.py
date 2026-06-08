from __future__ import annotations

import heapq
import random
from dataclasses import dataclass

from .config import RequestGenerationConfig
from .models import Request


@dataclass
class SessionState:
    session_id: int
    total_turns: int
    first_arrival_time_ms: float
    initial_prompt_tokens: int
    accumulate_context_across_turns: bool
    followup_prompt_tokens: int
    output_tokens_min: int
    output_tokens_max: int
    initial_prompt_bucket: str
    cumulative_context_tokens: int = 0


class RequestGenerator:
    """Owns user-session creation and follow-up arrival progression."""

    def __init__(
        self,
        config: RequestGenerationConfig,
        rng: random.Random,
    ) -> None:
        self.config = config
        self.rng = rng
        self.session_states: dict[int, SessionState] = {}
        self.pending_arrivals: list[tuple[float, int, int]] = []
        self.next_request_id = 1
        self._total_request_count = 0

    @property
    def session_count(self) -> int:
        return len(self.session_states)

    @property
    def total_request_count(self) -> int:
        return self._total_request_count

    def initialize(self) -> None:
        session_states = self._build_session_states()
        self.session_states = {
            session_state.session_id: session_state for session_state in session_states
        }
        self._total_request_count = sum(
            session_state.total_turns for session_state in session_states
        )
        self.pending_arrivals = [
            (session_state.first_arrival_time_ms, session_state.session_id, 1)
            for session_state in session_states
        ]
        heapq.heapify(self.pending_arrivals)

    def has_pending_arrivals(self) -> bool:
        return bool(self.pending_arrivals)

    def next_arrival_time_ms(self) -> float | None:
        if not self.pending_arrivals:
            return None
        return self.pending_arrivals[0][0]

    def take_ready_requests(self, current_time_ms: float) -> list[Request]:
        ready_requests: list[Request] = []
        while self.pending_arrivals:
            arrival_time_ms, session_id, turn_index = self.pending_arrivals[0]
            if arrival_time_ms > current_time_ms:
                break
            heapq.heappop(self.pending_arrivals)
            session_state = self.session_states[session_id]
            ready_requests.append(
                self._build_request(
                    session_state=session_state,
                    turn_index=turn_index,
                    arrival_time_ms=arrival_time_ms,
                )
            )
        return ready_requests

    def on_request_finished(self, request: Request) -> None:
        if request.session_id is None:
            return
        session_state = self.session_states[request.session_id]
        session_state.cumulative_context_tokens += (
            request.new_prompt_tokens + request.output_tokens
        )
        if request.turn_index >= session_state.total_turns:
            return
        heapq.heappush(
            self.pending_arrivals,
            (
                (request.finish_time_ms or 0.0) + self._sample_followup_gap_ms(),
                request.session_id,
                request.turn_index + 1,
            ),
        )

    def _build_session_states(self) -> list[SessionState]:
        cfg = self.config
        if cfg.new_user_arrival_mean_seconds <= 0:
            raise ValueError("new_user_arrival_mean_seconds must be positive.")
        if cfg.min_turns_per_user <= 0 or cfg.max_turns_per_user < cfg.min_turns_per_user:
            raise ValueError("Turn range must be positive and ordered.")
        if cfg.followup_prompt_tokens < 0:
            raise ValueError("followup_prompt_tokens cannot be negative.")
        if not 0.0 <= cfg.short_context_probability <= 1.0:
            raise ValueError("short_context_probability must be between 0 and 1.")

        current_time_ms = 0.0
        session_states: list[SessionState] = []
        for session_id in range(1, cfg.user_count + 1):
            current_time_ms += self.rng.expovariate(
                1.0 / cfg.new_user_arrival_mean_seconds
            ) * 1000.0
            prompt_bucket = (
                "short"
                if self.rng.random() < cfg.short_context_probability
                else "long"
            )
            output_tokens_min, output_tokens_max = self._output_range_for_bucket(
                prompt_bucket
            )
            session_states.append(
                SessionState(
                    session_id=session_id,
                    total_turns=self.rng.randint(
                        cfg.min_turns_per_user,
                        cfg.max_turns_per_user,
                    ),
                    first_arrival_time_ms=current_time_ms,
                    initial_prompt_tokens=self._sample_initial_prompt_tokens(prompt_bucket),
                    accumulate_context_across_turns=cfg.accumulate_context_across_turns,
                    followup_prompt_tokens=cfg.followup_prompt_tokens,
                    output_tokens_min=output_tokens_min,
                    output_tokens_max=output_tokens_max,
                    initial_prompt_bucket=prompt_bucket,
                )
            )
        return session_states

    def _sample_initial_prompt_tokens(self, prompt_bucket: str) -> int:
        cfg = self.config
        base_tokens = (
            cfg.short_context_prompt_tokens
            if prompt_bucket == "short"
            else cfg.long_context_prompt_tokens
        )
        variation = base_tokens * cfg.initial_prompt_variation_ratio
        min_tokens = max(1, int(round(base_tokens - variation)))
        max_tokens = max(min_tokens, int(round(base_tokens + variation)))
        return self.rng.randint(min_tokens, max_tokens)

    def _output_range_for_bucket(self, prompt_bucket: str) -> tuple[int, int]:
        cfg = self.config
        if prompt_bucket == "short":
            return (
                cfg.short_context_output_tokens_min,
                cfg.short_context_output_tokens_max,
            )
        return (
            cfg.long_context_output_tokens_min,
            cfg.long_context_output_tokens_max,
        )

    def _build_request(
        self,
        session_state: SessionState,
        turn_index: int,
        arrival_time_ms: float,
    ) -> Request:
        if not session_state.accumulate_context_across_turns:
            history_tokens = 0
            new_prompt_tokens = session_state.initial_prompt_tokens
        elif turn_index == 1:
            history_tokens = session_state.cumulative_context_tokens
            new_prompt_tokens = session_state.initial_prompt_tokens
        else:
            history_tokens = session_state.cumulative_context_tokens
            new_prompt_tokens = session_state.followup_prompt_tokens
        output_tokens = self.rng.randint(
            session_state.output_tokens_min,
            session_state.output_tokens_max,
        )
        request = Request(
            request_id=self.next_request_id,
            arrival_time_ms=arrival_time_ms,
            prompt_tokens=history_tokens + new_prompt_tokens,
            output_tokens=output_tokens,
            session_id=session_state.session_id,
            turn_index=turn_index,
            total_turns=session_state.total_turns,
            initial_prompt_bucket=session_state.initial_prompt_bucket,
            history_tokens=history_tokens,
            new_prompt_tokens=new_prompt_tokens,
        )
        self.next_request_id += 1
        return request

    def _sample_followup_gap_ms(self) -> float:
        cfg = self.config
        if cfg.followup_arrival_std_seconds < 0:
            raise ValueError("followup_arrival_std_seconds cannot be negative.")
        gap_seconds = self.rng.gauss(
            cfg.followup_arrival_mean_seconds,
            cfg.followup_arrival_std_seconds,
        )
        return max(0.0, gap_seconds * 1000.0)
