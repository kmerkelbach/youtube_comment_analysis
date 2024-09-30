from enum import Enum, auto


class ClassificationType(Enum):
    Sentiment = auto()
    Toxicity = auto()
    Emotion = auto()


classification_length_limits = {
    ClassificationType.Sentiment: 1500,
    ClassificationType.Toxicity: 1500,
    ClassificationType.Emotion: 500
}
