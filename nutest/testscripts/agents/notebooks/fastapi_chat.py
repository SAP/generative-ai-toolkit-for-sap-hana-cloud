from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from gen_ai_hub.proxy.langchain import init_llm
from hana_ai.agents.deprecated.scenario_agents import HANAChatAgent
from hana_ml import dataframe

code_llm = init_llm('gpt-4', temperature=0.0, max_tokens=2000) # used to do logical reasoning
chat_llm = init_llm('gpt-35-turbo', temperature=0.0, max_tokens=2000) #used to summarize history chat, refine response

url, port, user, pwd = "<HANA_HOST>.hana.demo-hc-3-haas-hc-dev.dev-aws.hanacloud.ondemand.com", 443, "<HANA_USER>", "<HANA_PASSWORD>"
connection_context = dataframe.ConnectionContext(url, port, user, pwd)

agent = HANAChatAgent(chat_llm=chat_llm,
                      code_llm=code_llm,
                      connection_context=connection_context,
                      verbose=True,
                      show_prompt=True,
                      force=True,
                      execute_confirm=False)

class Item(BaseModel):
    query: str

app = FastAPI()

@app.post("/chat")
async def chat(item: Item):
    return agent.chat(item.query)

uvicorn.run(app, host="0.0.0.0", port=5566)