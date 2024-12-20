import pytest
from langgraph.graph import StateGraph
from llm_chatbot_for_messengers.domain.chatbot import ChatbotState, Workflow, WorkflowNode
from llm_chatbot_for_messengers.domain.specification import WorkflowNodeSpecification, WorkflowSpecification
from pydantic import BaseModel


@pytest.mark.asyncio
async def test_workflow_node_spec():
    # given
    spec = WorkflowNodeSpecification[ChatbotState, BaseModel, ChatbotState](
        initial_schema=ChatbotState,
        final_schemas=(BaseModel, ChatbotState),
        name='test_node',
        func=lambda x, y, z: (x, y, z),
    )
    node = WorkflowNode[ChatbotState, BaseModel, ChatbotState](
        initial_schema=ChatbotState,
        final_schemas=(BaseModel, ChatbotState),
        name='test_node',
        func=lambda x, y, z: (x, y, z),
    )
    # then
    assert await spec.is_satisfied_by(node)


@pytest.mark.asyncio
async def test_workflow_spec():
    # given
    start_node_spec = WorkflowNodeSpecification[ChatbotState, ChatbotState](
        initial_schema=ChatbotState,
        final_schemas=(ChatbotState,),
        name='start',
        func=lambda x, y, z: (x, y, z),
    )
    end_node_spec = WorkflowNodeSpecification[ChatbotState, ChatbotState](
        initial_schema=ChatbotState,
        final_schemas=(ChatbotState,),
        name='end',
        func=lambda x, y, z: (x, y, z),
    )
    start_node_spec.add_children(end_node_spec)

    workflow_spec = WorkflowSpecification(
        start_node_spec=start_node_spec,
        end_node_spec=end_node_spec,  # type: ignore
    )
    # when
    start_node = WorkflowNode[ChatbotState, ChatbotState](
        initial_schema=ChatbotState,
        final_schemas=(ChatbotState,),
        name='start',
        func=lambda x, y, z: (x, y, z),
    )
    end_node = WorkflowNode[ChatbotState, ChatbotState](
        initial_schema=ChatbotState,
        final_schemas=(ChatbotState,),
        name='end',
        func=lambda x, y, z: (x, y, z),
    )
    start_node.add_children(end_node)
    workflow = Workflow(start_node=start_node, end_node=end_node, graph=StateGraph(ChatbotState).compile())  # type: ignore
    # then
    assert await workflow_spec.is_satisfied_by(workflow)
