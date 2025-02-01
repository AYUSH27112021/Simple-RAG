import operator
from langchain_ollama import ChatOllama
from typing_extensions import TypedDict
from typing import List, Annotated
# from langchain.schema import Document
from langgraph.graph import END
from Embed_store import get_retriever
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph
import json

class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """

    question: str
    generation: str
    max_retries: int
    answers: int
    loop_step: Annotated[int, operator.add]
    documents: List[str]

class GraphRag:
    def __init__(self):
        local_llm = "llama3.2:3b-instruct-fp16"
        self.llm = ChatOllama(model=local_llm, temperature=0)
        self.llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")

    def retrieve(self,state):
        """
        Retrieve documents from vectorstore
        """
        print("---RETRIEVE---")
        question = state["question"]
        # Write retrieved documents to documents key in state
        retriever=get_retriever()
        documents = retriever.invoke(question)
        return {"documents": documents}

    def generate(self,state):
        """
        Generate answer using RAG on retrieved documents
        """
        rag_prompt = """You are an assistant for question-answering tasks. 
        Here is the context to use to answer the question:
        {context} 
        Think carefully about the above context. 
        Now, review the user question:
        {question}
        Provide an answer to this questions using only the above context. 
        Use three sentences maximum and keep the answer concise.
        Answer:"""

        # Post-processing
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        loop_step = state.get("loop_step", 0)

        # RAG generation
        docs_txt = format_docs(documents)
        rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
        generation = self.llm.invoke([HumanMessage(content=rag_prompt_formatted)])
        return {"generation": generation, "loop_step": loop_step + 1}

    def grade_documents(self,state):
        """
        Determines whether the retrieved documents are relevant to the question
        """
        # Doc grader instructions
        doc_grader_instructions = """You are a grader assessing relevance of a retrieved document 
        to a user question.If the document contains keyword(s) or semantic meaning related to 
        the question, grade it as relevant."""

        # Grader prompt
        doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the 
        user question: \n\n {question}.This carefully and objectively assess whether the document 
        contains at least some information that is relevant to the question.Return JSON with single 
        key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains 
        at least some information that is relevant to the question."""

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        for d in documents:
            doc_grader_prompt_formatted = doc_grader_prompt.format(
                document=d.page_content, question=question
            )
            result = self.llm_json_mode.invoke(
                [SystemMessage(content=doc_grader_instructions)]
                + [HumanMessage(content=doc_grader_prompt_formatted)]
            )
            grade = json.loads(result.content)["binary_score"]
            # Document relevant
            if grade.lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
        return {"documents": filtered_docs}

    def decide_to_generate(self,state):
        """
        Determines whether to generate an answer
        """
        print("---ASSESS GRADED DOCUMENTS---")
        question = state["question"]
        filtered_documents = state["documents"]

        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

    def grade_generation_v_documents_and_question(self,state):
        """
        Determines whether the generation is grounded in the document and answers question
        """
        hallucination_grader_instructions = """You are a teacher grading a quiz. You will be given FACTS and a 
        STUDENT ANSWER. Here is the grade criteria to follow:
        (1) Ensure the STUDENT ANSWER is grounded in the FACTS.
        (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.
        Score:
        A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 
        A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible
        score you can give.Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion 
        are correct. Avoid simply stating the correct answer at the outset."""

        # Grader prompt
        hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}. 
        Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the 
        STUDENT ANSWER is grounded in the FACTS. And a key, explanation, that contains an explanation 
        of the score."""

        # Answer grader instructions
        answer_grader_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a STUDENT 
        ANSWER. Here is the grade criteria to follow:(1) The STUDENT ANSWER helps to answer the QUESTION
        Score:A score of yes means that the student's answer meets all of the criteria. This is the highest 
        (best) score. The student can receive a score of yes if the answer contains extra information that is
        not explicitly asked for in the question.A score of no means that the student's answer does not meet all 
        of the criteria. This is the lowest possible score you can give.Explain your reasoning in a step-by-step 
        manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the 
        outset."""

        # Grader prompt
        answer_grader_prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}. Return JSON with two 
        two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the criteria. 
        And a key, explanation, that contains an explanation of the score."""

        # Post-processing
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        max_retries = state.get("max_retries", 3)  # Default to 3 if not provided

        hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
            documents=format_docs(documents), generation=generation.content
        )
        result = self.llm_json_mode.invoke(
            [SystemMessage(content=hallucination_grader_instructions)]
            + [HumanMessage(content=hallucination_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]

        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            # Test using question and generation from above
            answer_grader_prompt_formatted = answer_grader_prompt.format(
                question=question, generation=generation.content
            )
            result = self.llm_json_mode.invoke(
                [SystemMessage(content=answer_grader_instructions)]
                + [HumanMessage(content=answer_grader_prompt_formatted)]
            )
            grade = json.loads(result.content)["binary_score"]
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            elif state["loop_step"] <= max_retries:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
            else:
                print("---DECISION: MAX RETRIES REACHED---")
                return "max retries"
        elif state["loop_step"] <= max_retries:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"
        else:
            print("---DECISION: MAX RETRIES REACHED---")
            return "max retries"
        
    def invoke_graph(self,inputs):
        workflow = StateGraph(GraphState)
        # Define the nodes
        workflow.add_node("retrieve", self.retrieve)  # retrieve
        workflow.add_node("grade_documents", self.grade_documents)  # grade documents
        workflow.add_node("generate", self.generate)  # generate

        # Build graph
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")

        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "generate": "generate",
                "no relevant docs": END,
            },
        )

        max_retries = 2
        workflow.add_conditional_edges(
            "generate",
            self.grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "retrieve" if max_retries > 0 else END,
                "max retries": END,
            },
        )
        # Compile
        graph = workflow.compile()
        # Execute the workflow
        for event in graph.stream(inputs, stream_mode="values"):
            if "generation" in event and isinstance(event["generation"], str):
                return event["generation"].content