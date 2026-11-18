"""Which counter feeds cpu_cores (check 21 of the workers task)."""

from chordfinder.load import ServerStats, server_stats


def text(**values: float) -> str:
    return "".join(f"# TYPE {k} counter\n{k} {v}\n" for k, v in values.items())


def test_own_counter_wins_over_process_cpu() -> None:
    before: str = text(chordfinder_cpu_seconds_total=10, process_cpu_seconds_total=10)
    after: str = text(chordfinder_cpu_seconds_total=14, process_cpu_seconds_total=12)
    stats: ServerStats = server_stats(before, after, 4.0)
    assert stats.cpu_cores == 1.0


def test_own_counter_alone_is_enough() -> None:
    stats: ServerStats = server_stats(
        text(chordfinder_cpu_seconds_total=10), text(chordfinder_cpu_seconds_total=14), 4.0
    )
    assert stats.cpu_cores == 1.0


def test_own_counter_in_one_snapshot_falls_back_to_process_cpu() -> None:
    before: str = text(process_cpu_seconds_total=10)
    after: str = text(chordfinder_cpu_seconds_total=14, process_cpu_seconds_total=12)
    stats: ServerStats = server_stats(before, after, 4.0)
    assert stats.cpu_cores == 0.5
