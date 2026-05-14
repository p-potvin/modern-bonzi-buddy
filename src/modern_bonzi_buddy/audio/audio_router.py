from __future__ import annotations


class AudioRouter:
    """Placeholder for machine audio feed routing and playback plumbing."""

    def __init__(self) -> None:
        self.running = False

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False
