from llm_chatbot_for_messengers.ioc_container.container import get_qa_agent
from llm_chatbot_for_messengers.messenger_if.kakao.api import app
from llm_chatbot_for_messengers.messenger_if.kakao.vo import ChatResponse


def test_chat(client, fake_agent):
    app.dependency_overrides[get_qa_agent] = lambda: fake_agent
    response = client.post(
        '/kakao/v1/chat',
        json={
            'userRequest': {
                'utterance': '오늘 점심 뭐 먹을까?',
                'user': {
                    'id': 'test',
                    'type': 'swagger',
                },
            },
        },
    )

    assert response.status_code == 200
    assert ChatResponse(**response.json())


def test_failed_chat(client, fake_agent_fallback):
    app.dependency_overrides[get_qa_agent] = lambda: fake_agent_fallback
    response = client.post(
        '/kakao/v1/chat',
        json={
            'userRequest': {
                'utterance': '오늘 점심 뭐 먹을까?',
                'user': {
                    'id': 'test',
                    'type': 'swagger',
                },
            },
        },
    )

    assert response.status_code == 200
    response = ChatResponse(**response.json())
    assert response.template.outputs[0].simpleText.text == 'Fallback message'
