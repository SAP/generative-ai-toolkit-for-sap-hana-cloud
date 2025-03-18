"""
This module uses llm to grade the retrieved doc and find the most relevant ones.

The following class is available:

    * :class `CorrectiveRetriever`
"""

# pylint: disable=invalid-name
# pylint: disable=misplaced-bare-raise
# pylint: disable=logging-fstring-interpolation

import logging
from typing import TypedDict, Dict

from langgraph.graph import END, StateGraph
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.utils.function_calling import convert_to_openai_tool

logger = logging.getLogger(__name__) #pylint: disable=invalid-name

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """

    keys: Dict[str, any]

class CorrectiveRetriever(object):
    """
    Corrective retriever class.

    Parameters:
    -----------
    vectordb: any
        Vector database.
    llm: any
        LLM.
    max_iter: int
        Maximum iterations.
    recursion_limit: int
        Recursion limit.
    """
    vectordb: any
    llm: any
    max_iter: int
    recursion_limit: int
    workflow: StateGraph
    def __init__(self, vectordb, llm, max_iter=3, recursion_limit=100):
        """
        Init corrective retriever.
        """
        self.vectordb = vectordb
        self.max_iter = max_iter
        self.llm = llm
        self.recursion_limit = recursion_limit
        self.workflow = StateGraph(GraphState)

    def _retrieve(self, state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        logger.info("---RETRIEVE---")
        state_dict = state["keys"]
        question = state_dict["question"]
        top_k = state_dict["top_k"]
        init_k = state_dict["init_k"]
        documents = self.vectordb.query(input=question, top_n=init_k)
        return {"keys": {"documents": documents, "question": question, "top_k": top_k, "init_k": init_k}}

    def _grade_documents(self, state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with relevant documents
        """

        logger.info("---CHECK RELEVANCE---")
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]
        top_k = state_dict["top_k"]
        init_k = state_dict["init_k"]

        # Data model
        class grade(BaseModel):
            """Binary score for relevance check."""

            binary_score: str = Field(description="Relevance score 'yes' or 'no'")

        # LLM
        model = self.llm

        # Tool
        grade_tool_oai = convert_to_openai_tool(grade)

        # LLM with tool and enforce invocation
        llm_with_tool = model.bind(
            tools=[grade_tool_oai],
            tool_choice={"type": "function", "function": {"name": "grade"}},
        )

        # Parser
        parser_tool = PydanticToolsParser(tools=[grade])

        # Prompt
        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
            input_variables=["context", "question"],
        )

        # Chain
        chain = prompt | llm_with_tool | parser_tool

        # Score
        search = "No"  # Default do not opt for second search to supplement retrieval
        score = chain.invoke({"question": question, "context": documents})
        grade = score[0].binary_score
        if grade == "yes":
            logger.info("---GRADE: DOCUMENT RELEVANT---")
            pass
        else:
            logger.info("---GRADE: DOCUMENT NOT RELEVANT---")
            search = "Yes"  # Perform second search
            init_k = init_k + 1

        if init_k > top_k:
            logger.info("exceed the maximum iterations!")
            raise

        return {
            "keys": {
                "documents": documents,
                "question": question,
                "top_k" : top_k,
                "init_k" : init_k,
                "run_second_search": search,
            }
        }

    def _generate(self, state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        logger.info("---GENERATE---")
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]
        top_k = state_dict["top_k"]
        init_k = state_dict["init_k"]
        return {
            "keys": {"documents": documents, "question": question, "top_k": top_k, "init_k": init_k, "generation": documents}
        }

    def _decide_to_generate(self, state):
        """
        Determines whether to generate an answer or re-generate a question for second search.

        Args:
            state (dict): The current state of the agent, including all keys.

        Returns:
            str: Next node to call
        """

        logger.info("---DECIDE TO GENERATE---")
        state_dict = state["keys"]
        # question = state_dict["question"]
        # documents = state_dict["documents"]
        # top_k = state_dict["top_k"]
        # init_k = state_dict["init_k"]
        search = state_dict["run_second_search"]

        if search == "Yes":
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            logger.info("---DECISION: RUN THE SECOND SEARCH---")
            return "retrieve"
        else:
            # We have relevant documents, so generate answer
            logger.info("---DECISION: GENERATE---")
            return "generate"

    def query(self, query):
        """
        Query knowledge.

        Parameters:
        -----------
        query: str
            Query.
        """
        workflow = self.workflow

        # Define the nodes
        workflow.add_node("retrieve", self._retrieve)  # retrieve
        workflow.add_node("grade_documents", self._grade_documents)  # grade documents
        workflow.add_node("generate", self._generate)  # generatae

        # Build graph
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self._decide_to_generate,
            {
                "retrieve": "retrieve",
                "generate": "generate",
            },
        )
        workflow.add_edge("generate", END)

        # Compile
        app = workflow.compile()
        # Correction for question not present in context
        inputs = {
            "keys": {
                "question": query,
                "top_k" : self.max_iter,
                "init_k" : 1
            }
        }
        result = None
        for output in app.stream(inputs,{"recursion_limit": self.recursion_limit}):
            for key, value in output.items():
                # Node
                logger.info(f"Node '{key}':")
                # Optional: print full state
                logger.info(value["keys"], indent=2, width=80, depth=None)
                result = value
            logger.info("\n---\n")


        # Final generation
        logger.info(result["keys"]["generation"])
        return result["keys"]["generation"]
