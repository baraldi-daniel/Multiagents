from langchain_core.messages import HumanMessage
from graph.graph import graph






def invoke_app(prompt,id_session):
    user_id = "1"
    config = {"configurable": {"thread_id": f"{id_session}", "user_id": user_id}}
    return graph.invoke(input={"messages": [HumanMessage(prompt)]}, config=config)