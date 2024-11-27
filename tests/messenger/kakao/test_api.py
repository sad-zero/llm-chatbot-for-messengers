from llm_chatbot_for_messengers.messenger.kakao.vo import ChatResponse


def test_chat(client):
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


def test_failed_chat(failed_client):
    response = failed_client.post(
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
