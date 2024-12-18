# Design(Current)
The project's architecture refers to **Domain Driven Design**.

## Ubiquitous Language
| Concept   | Description                                                 |
| --------- | ----------------------------------------------------------- |
| Chatbot   | Service to answer various questions in the given time limit |
| Messenger | Text-based communication channel                            |
| User      | Messenger's user                                            |
| Memory    | Chat histories between a chatbot and an user                |
| Workflow  | Chatbot's senario                                           |
| Template  | LLM's prompt template                                       |

## Senario
### Initialize chatbot
```mermaid
sequenceDiagram
    autonumber
    actor Start
    Start ->>+ Agent: Initialize with AgentConfig
    Agent ->>+ MemoryManager: Intialize memory
    MemoryManager -->>- Agent: Initialized memory
    Agent ->>+ Workflow: Intialize with WorkflowConfig and memory
    Workflow ->>+ Template: Retrieve prompts
    Template ->>+ Parser: Convert prompts to valid instance
    Parser -->>- Template: Prompt instance
    Template -->>- Workflow: Prompt instance
    Workflow -->>- Agent: Initialized workflow
    Agent -->>- Start: Initialized agent
```
### User ask a question

```mermaid
sequenceDiagram
    autonumber
    box Messenger
        actor User
    end
    box MessengerIF
        participant MessengerApi
        participant Middleware
    end
    box Chatbot
        participant Dao
        participant Agent
        participant Workflow
        participant Memory
    end
    participant LLM

    User ->>+ MessengerApi: Ask a question
    MessengerApi ->>+ Middleware: Verify user's request
    Middleware ->>+ Dao: Retrieve messenger's information
    Dao -->>- Middleware: Messenger
    Middleware -->>- MessengerApi: Validated request
    MessengerApi ->>+ Agent: Delegate the question to agent
    Agent ->>+ Workflow: Delegate the question to workflow
    Workflow ->>+ Memory: Retrieve previous chat histories
    Memory -->>- Workflow: Chat histories
    Workflow ->>+ LLM: Generate answer
    LLM -->>- Workflow: Return the answer
    Workflow -->>- Agent: Return the answer
    Agent -->>- MessengerApi: Return the answer
    MessengerApi -->>- User: answer the question
```

## Project Structure

- `src/llm_chatbot_for_messengers/`
    - `core/`: Core models package
        - `entity/`: Entities package
            - `agent`: Represent chatbot
            - `messenger`: Represent messengers
            - `user`: Represent messenger's users
        - `output/`: Outbound services package
            - `dao`: Manage persistances
            - `memory`: Manage agent's memory
            - `parser`: Convert LLM prompt template file to object
            - `template`: Manage LLM prompt template store
        - `workflow/`: Chatbot's workflow
            - `base`: Abstract workflow
            - `qa`: QA Workflow
            - `vo`: Workflow's Value objects
        - `configuration`: Chatbot's configuration
        - `error`: Core package's errors
        - `specificatgion`: Constraints of entites and vos
        - `vo`: Value objects of Entities
    - `messengers/`: Messenger's custom chatbot IF package
        - `kakao/`: Kakao custom chatbot IF
        - `middleware/`: Common middlewares
        - `vo`: Common Value objects
    - `main`: Project's entrypoint.
