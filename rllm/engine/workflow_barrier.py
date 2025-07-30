import asyncio
import random
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any

from rllm.workflows.workflow import TerminationEvent, TerminationReason


class WorkflowBarrier:
    """A synchronization barrier for coordinating multiple async workflows by episode UID."""

    # Note: doesn't support syncronizing workflows excuting the same task on different machines
    # fine as long as tasks aren't shared across machines.

    def __init__(self, episode_uids: list[str], min_peers: int = 1, max_peers: int | None = None) -> None:
        """Initialize the barrier with a set of episode UIDs and peer constraints.

        Args:
            episode_uids: List of episode UIDs to synchronize.
            min_peers: Minimum number of peers (excluding self) required to proceed.
            max_peers: Maximum number of peers (excluding self) to return in wait().
        """
        self.min_peers = min_peers
        self.max_peers = max_peers
        self.reset(episode_uids)

    def reset(self, episode_uids: list[str]) -> None:
        """Reset the barrier for a new set of episode UIDs."""
        self.task_id_to_uids: dict[str, set[str]] = defaultdict(set)
        for uid in episode_uids:
            task_id = uid.split("_")[0]
            self.task_id_to_uids[task_id].add(uid)
        self._barrier_states: dict[str, dict[str, Any]] = {}
        self._barrier_events: dict[str, asyncio.Event] = defaultdict(asyncio.Event)
        self._barrier_waiters: dict[str, int] = defaultdict(int)

    @asynccontextmanager
    async def wait(
        self,
        uid: str,
        state: Any,
        rng: random.Random | None = None,
    ):
        """
        Wait for all workflows of this task_id to arrive, then yield a list of randomly selected peer state dicts (excluding self).

        Args:
            uid: This workflow's episode uid.
            state: The state to share at the barrier.
            rng: Optional random.Random instance for reproducibility.

        Yields:
            peer_states: List[Any] for the selected peers (excluding self), in random order.
        """
        if not self.task_id_to_uids:
            raise ValueError("Barrier not initialized")

        task_id = uid.split("_")[0]
        if uid not in self.task_id_to_uids[task_id]:
            raise ValueError(f"UID {uid} not expected for task_id {task_id}")

        n_expected = len(self.task_id_to_uids[task_id])
        # min_peers is the minimum number of peers (excluding self)
        if n_expected - 1 < self.min_peers:
            raise TerminationEvent(TerminationReason.NOT_ENOUGH_PEERS)
        state_dict = self._barrier_states.setdefault(task_id, {})
        state_dict[uid] = state
        self._barrier_waiters[task_id] += 1
        event = self._barrier_events[task_id]

        # If all expected have arrived, release the barrier
        if self._barrier_waiters[task_id] == n_expected:
            event.set()
        else:
            await event.wait()

        # After the barrier is released, re-check if enough peers remain
        n_expected = len(self.task_id_to_uids[task_id])
        if n_expected - 1 < self.min_peers:
            raise TerminationEvent(TerminationReason.NOT_ENOUGH_PEERS)

        # Print total number of workflows still running for this task_id
        n_total = len(self.task_id_to_uids[task_id])
        print(f"[Barrier] Task group '{task_id}' released: {n_total} workflows still running for this task.")

        # Select peer states (excluding self)
        peer_uids = [x for x in state_dict if x != uid]
        if self.max_peers is not None and self.max_peers < len(peer_uids):
            rng = rng or random
            selected_uids = rng.sample(peer_uids, self.max_peers)
        else:
            selected_uids = peer_uids
        peer_states = [state_dict[k] for k in selected_uids]

        try:
            yield peer_states
        finally:
            # Cleanup after all have passed the barrier
            self._barrier_waiters[task_id] -= 1
            if self._barrier_waiters[task_id] == 0:
                self._barrier_waiters.pop(task_id, None)
                self._barrier_states.pop(task_id, None)
                self._barrier_events.pop(task_id, None)

    def mark_terminated(self, uid: str) -> None:
        """
        Mark a rollout as terminated so the barrier will not expect it to arrive.
        This allows other rollouts to proceed if all remaining expected arrivals are met,
        or if the number of remaining agents drops below min_peers.
        """
        if not self.task_id_to_uids:
            return
        task_id = uid.split("_")[0]
        if uid in self.task_id_to_uids.get(task_id, set()):
            self.task_id_to_uids[task_id].remove(uid)
        if task_id in self._barrier_states and uid in self._barrier_states[task_id]:
            del self._barrier_states[task_id][uid]
        n_expected = len(self.task_id_to_uids.get(task_id, set()))
        n_arrived = len(self._barrier_states.get(task_id, {}))
        # Release the barrier if not enough peers remain, or if all have arrived
        if (n_expected - 1 < self.min_peers) or (n_expected > 0 and n_arrived == n_expected):
            self._barrier_events[task_id].set()
