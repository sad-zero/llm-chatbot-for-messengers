import pytest
from llm_chatbot_for_messengers.core.vo import UserId
from pydantic import ValidationError


@pytest.mark.parametrize(
    ('user_seq', 'user_id', 'expected'),
    [(1, 'test_id', 'ok'), (1, None, 'ok'), (None, 'test_id', 'ok'), (None, None, 'error')],
)
def test_user_id(user_seq: int, user_id: str, expected: str):
    match expected:
        case 'ok':
            assert UserId(user_seq=user_seq, user_id=user_id)
        case 'error':
            with pytest.raises(ValidationError):
                UserId(user_seq=user_seq, user_id=user_id)
