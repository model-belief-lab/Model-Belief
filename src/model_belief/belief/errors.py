class BeliefError(Exception):
    """Base error for belief module."""


class BeliefInputError(BeliefError):
    """Invalid inputs / missing required fields."""


class BeliefParseError(BeliefError):
    """Unable to parse provider logprobs structure into expected form."""