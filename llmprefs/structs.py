import enum

from pydantic import BaseModel

TaskId = int


class TaskTopic(enum.Enum):
    pass


class TaskType(enum.Enum):
    pass


class TaskDifficulty(enum.Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class TaskImpact(enum.Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"


class TaskRecord(BaseModel):
    """Task plus categorical metadata."""

    id: TaskId
    task: str
    topic: TaskTopic
    type: TaskType
    difficulty: TaskDifficulty
    impact: TaskImpact
