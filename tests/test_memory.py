"""Tests for working memory module."""


from src.memory.working import WorkingMemory


class TestWorkingMemory:
    """Tests for WorkingMemory sliding window behavior."""

    def test_empty_memory_returns_only_system_prompt(self) -> None:
        memory = WorkingMemory(system_prompt="You are helpful.")
        snapshot = memory.get_snapshot()
        messages = snapshot.to_message_list()

        assert len(messages) == 1
        assert messages[0].role == "system"
        assert messages[0].content == "You are helpful."

    def test_empty_memory_no_system_prompt(self) -> None:
        memory = WorkingMemory()
        snapshot = memory.get_snapshot()

        assert snapshot.system_message is None
        assert snapshot.to_message_list() == []

    def test_add_single_turn(self) -> None:
        memory = WorkingMemory(system_prompt="System.")
        memory.add_user_message("Hello")
        memory.add_assistant_message("Hi there!")

        snapshot = memory.get_snapshot()
        messages = snapshot.to_message_list()

        assert len(messages) == 3
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert messages[1].content == "Hello"
        assert messages[2].role == "assistant"
        assert messages[2].content == "Hi there!"

    def test_turn_count_tracks_user_messages(self) -> None:
        memory = WorkingMemory()
        assert memory.turn_count == 0

        memory.add_user_message("First")
        assert memory.turn_count == 1

        memory.add_assistant_message("Response")
        assert memory.turn_count == 1

        memory.add_user_message("Second")
        assert memory.turn_count == 2

    def test_sliding_window_trims_oldest_pairs(self) -> None:
        memory = WorkingMemory(max_turns=2)

        memory.add_user_message("Turn 1")
        memory.add_assistant_message("Response 1")
        memory.add_user_message("Turn 2")
        memory.add_assistant_message("Response 2")
        memory.add_user_message("Turn 3")
        memory.add_assistant_message("Response 3")

        assert memory.turn_count == 2
        snapshot = memory.get_snapshot()
        messages = snapshot.to_message_list()

        assert messages[0].content == "Turn 2"
        assert messages[1].content == "Response 2"
        assert messages[2].content == "Turn 3"
        assert messages[3].content == "Response 3"

    def test_system_prompt_preserved_after_trim(self) -> None:
        memory = WorkingMemory(max_turns=1, system_prompt="Keep me.")

        memory.add_user_message("Turn 1")
        memory.add_assistant_message("Response 1")
        memory.add_user_message("Turn 2")
        memory.add_assistant_message("Response 2")

        messages = memory.get_snapshot().to_message_list()

        assert messages[0].role == "system"
        assert messages[0].content == "Keep me."
        assert len(messages) == 3  # system + 1 user + 1 assistant

    def test_reset_clears_history_preserves_system(self) -> None:
        memory = WorkingMemory(system_prompt="Persist.")
        memory.add_user_message("Hello")
        memory.add_assistant_message("Hi")
        memory.reset()

        assert memory.turn_count == 0
        assert memory.message_count == 0
        messages = memory.get_snapshot().to_message_list()
        assert len(messages) == 1
        assert messages[0].content == "Persist."

    def test_snapshot_is_immutable(self) -> None:
        memory = WorkingMemory()
        memory.add_user_message("Hello")
        snapshot = memory.get_snapshot()

        memory.add_assistant_message("Hi")
        memory.add_user_message("More")

        assert len(snapshot.messages) == 1

    def test_message_count(self) -> None:
        memory = WorkingMemory(system_prompt="System.")
        memory.add_user_message("A")
        memory.add_assistant_message("B")

        assert memory.message_count == 2

    def test_trim_returns_evicted_messages(self) -> None:
        memory = WorkingMemory(max_turns=1)

        memory.add_user_message("Turn 1")
        memory.add_assistant_message("Response 1")

        evicted = memory.add_user_message("Turn 2")

        assert len(evicted) == 2
        assert evicted[0].content == "Turn 1"
        assert evicted[1].content == "Response 1"

    def test_no_eviction_when_under_limit(self) -> None:
        memory = WorkingMemory(max_turns=5)

        evicted = memory.add_user_message("Turn 1")
        assert evicted == []

    def test_summary_included_in_snapshot(self) -> None:
        memory = WorkingMemory(system_prompt="System.")
        memory.summary = "User discussed Python earlier."
        memory.add_user_message("Hello")

        messages = memory.get_snapshot().to_message_list()

        assert len(messages) == 3
        assert messages[0].role == "system"
        assert messages[0].content == "System."
        assert messages[1].role == "system"
        assert "User discussed Python earlier." in messages[1].content
        assert messages[2].role == "user"

    def test_reset_clears_summary(self) -> None:
        memory = WorkingMemory()
        memory.summary = "Some context."
        memory.reset()
        assert memory.summary == ""
