import asyncio
import time

import pytest
from llm_chatbot_for_messengers.core.error import SpecificationError
from llm_chatbot_for_messengers.core.specification import check_timeout


def _normal_func(elapsed: int):
    time.sleep(elapsed)
    return True


async def _async_func(elapsed: int):
    await asyncio.sleep(elapsed)
    return True


@pytest.mark.parametrize(
    ('elapsed', 'timeout', 'expected'),
    [
        (1.7, 2, 'ok'),
        (2, 2, 'error'),
        (2.1, 2, 'error'),
        (3.1, 2, 'error'),
    ],
)
def test_check_timeout_normal(elapsed, timeout, expected):
    traced = check_timeout(func=_normal_func, timeout=timeout)
    match expected:
        case 'ok':
            assert traced(elapsed)
        case 'error':
            s = time.time()
            with pytest.raises(SpecificationError):
                traced(elapsed)
            e = time.time()
            assert abs((e - s) - timeout) < 0.5


@pytest.mark.parametrize(
    ('elapsed', 'timeout', 'expected'),
    [
        (1.7, 2, 'ok'),
        (2, 2, 'error'),
        (2.1, 2, 'error'),
    ],
)
@pytest.mark.asyncio
async def test_check_timeout_async(elapsed, timeout, expected):
    traced = check_timeout(func=_async_func, timeout=timeout)
    match expected:
        case 'ok':
            assert await traced(elapsed)
        case 'error':
            with pytest.raises(SpecificationError):
                assert await traced(elapsed)
