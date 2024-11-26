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
