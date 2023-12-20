from langchain.schema.language_model import BaseLanguageModel
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import HumanMessagePromptTemplate
from langchain.chains.base import Chain
from langchain.pydantic_v1 import BaseModel
from .prompt import TEMPLATE
from typing import Optional

prompt = ChatPromptTemplate.from_messages([HumanMessagePromptTemplate.from_template(TEMPLATE)])

def get_instruction(question: str = None):
  if question is None:
    return ""
  
  return """
    Include any information that can be used to answer the question: "{question}". '
            "Do not directly answer the question itself.
  """.format(question=question)
    

class SummaryChainInput(BaseModel):
  question: Optional[str]
  text: str

# if you update this, you MUST also update ../pyproject.toml
# with the new `tool.langserve.export_attr`
def create_summary_chain(
    llm: BaseLanguageModel, 
  ) -> Chain:
  chain = (
    {
      'text': lambda x: x['text'],
      'instruction': lambda x: get_instruction(x['question'])
    }
    |
    prompt
    |
    llm
  )
  return chain
