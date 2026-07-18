"""Non-persisted Session result values."""

from dataclasses import dataclass

from myclaw.session.identifiers import require_session_id


@dataclass(frozen=True, slots=True)
class ResumeResult:
    """Identity of the Conversation Session selected by a successful resume."""

    session_id: str

    def __post_init__(self) -> None:
        require_session_id(self.session_id)
