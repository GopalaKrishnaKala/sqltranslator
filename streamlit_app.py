import json
from typing import TypedDict, Annotated, Union
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import os
import streamlit as st


st.sidebar.header("OpenAI API Key Configuration")
api_key = st.sidebar.text_input(
    "Enter your OpenAI API Key:",
    type="password",
    help="Your API key will only be used for this session and not stored anywhere.",
)

# Validate and set the API key
if api_key:
    os.environ['OPENAI_API_KEY'] = api_key
    OPENAI_API_KEY = api_key
else:
    st.warning("Please enter your OpenAI API key to proceed.")
    st.stop()

class ConverterState(TypedDict):
    input_query: str
    ast: Annotated[Union[dict, str, None], None]
    final_sql: Annotated[str, None]


llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-4o",  
    streaming=False
)

def parse_sql_to_ast(state: ConverterState) -> dict:

    with st.spinner("Thinking..."):
        query = state["input_query"]

        system_prompt = """
    Role: You are a SQL parsing assistant, and you are an expert at building Abstract Syntax Trees (ASTs) from Snowflake SQL.

    Task: Convert a single Snowflake SQL query into a valid JSON AST.

    Input Parameters:
     - A Snowflake SQL query as raw text.

    Step-by-Step Guidelines:
     1) Read the Snowflake SQL query thoroughly, noting any clauses like SELECT, FROM, JOIN, WHERE, GROUP BY, HAVING, ORDER BY, etc.
     2) Identify specialized Snowflake clauses or keywords (e.g., QUALIFY, ILIKE, ASOF JOIN, MATCH_CONDITION, etc.) and represent them in the JSON structure.
     3) For each clause or sub-expression, create a JSON key-value pair. For example:
        {
           "type": "select_statement",
           "select_list": [...],
           "from_clause": {...},
           "where_clause": {...},
           "order_by_clause": [...],
           ...
        }
     4) If the query contains multiple conditions, nest them in arrays or objects that reflect the logical structure.
     5) Include table aliases, function calls, and subqueries. For instance, if there's a subquery in the FROM clause, represent it as a nested object.
     6) Do not omit or rearrange essential elements, even if they seem unimportant. The AST must mirror the Snowflake query's logic.
     7) Avoid commentary or partial text. Output MUST be strictly valid JSON with no code fences, markdown, or explanation.
     8) If a portion of the query is ambiguous, choose a consistent JSON representation that preserves the query's intent.
     9) For example, you might have a nested structure like:
        {
           "type": "join_expression",
           "join_type": "ASOF",
           "left_table": {...},
           "right_table": {...},
           "match_condition": {...}
        }
        if the Snowflake query uses an ASOF JOIN with MATCH_CONDITION.
    10) Remain consistent in naming keys across the entire AST. For instance, if you call the main query node "select_statement", do not rename it to "query_statement" midway.
    11) Output only the JSON data structure, ensuring it can be directly parsed by a standard JSON parser.
    12) If you are unsure about certain Snowflake keywords, model them as logically as possible. For instance, treat ILIKE as a variant of a comparison operator, or store it under some "operator" key.
    13) Do not wrap your JSON in triple backticks (```), or any other code fence formatting.
    14) Do not prepend or append any text before or after the JSON. The final answer should be raw JSON.

    Output:
     - A strictly valid JSON AST reflecting all clauses in the input Snowflake SQL.

    Important Notes:
     - The final answer must be valid JSON with no syntax errors.
     - Be sure to capture subselects, join conditions, aliases, function calls, window functions, and ordering.
     - No additional explanation, commentary, or markdown is allowed.
     - Example minimal structure: { "type": "select_statement", "select_list": [...], "from_clause": {...} }
     - Remember that subsequent steps will rely on this AST for translation to ANSI SQL, so completeness and clarity are crucial.
     - If the query includes multiple statements, each statement should be reflected in the AST, though typically we handle one statement at a time.
     - End your output immediately after the closing brace of the JSON object—nothing else.
    End of prompt.
    """
        user_message = f"SQL to parse:\n{query}"

        response = llm.invoke(
            [ 
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
        )

        try:
            ast_data = json.loads(response.content)
        except Exception as e:
            ast_data = {"error": str(e), "raw_response": response.content}

        return {
            "ast": ast_data
        }

def translate_ast_to_ansi(state: ConverterState) -> dict:

    with st.spinner("Translating Snowflake SQL to ANSI SQL..."):
        ast_data = state["ast"]
        original_sql = state["input_query"]

        system_prompt = """
    Role: You are an expert SQL translator, specializing in converting Snowflake SQL to ANSI SQL.

    Task: Take two inputs:
      (1) the original Snowflake SQL,
      (2) the JSON AST derived from that query,
    and produce logically equivalent ANSI SQL.

    Input Parameters:
     - Original Snowflake SQL text.
     - JSON AST representing the structure of the same query.

    Step-by-Step Guidelines:
     1) Read the AST carefully to understand the Snowflake query structure.
     2) Check the original Snowflake SQL if the AST lacks detail or is ambiguous.
     3) Identify any Snowflake-specific features that may not exist in ANSI, such as:
        - ILIKE => use LOWER(column) LIKE LOWER(value)
        - QUALIFY => transform into a WHERE clause on a window function
        - ASOF JOIN or MATCH_CONDITION => emulate with window functions or correlated subqueries
        - TIME or DATE functions unique to Snowflake => approximate using standard SQL if possible
     4) Replace each Snowflake feature with ANSI-friendly logic. Preserve identical filters, ordering, grouping, etc.
     5) If the query references Snowflake UDFs or advanced syntax, replicate them or comment them out if there is no direct ANSI equivalent. Do not silently remove them.
     6) Pay close attention to unusual join types. If Snowflake uses LATERAL or ASOF, approximate them with standard joins or subqueries.
     7) Keep every column, alias, expression, and clause intact. Do not omit or rename columns arbitrarily.
     8) Observe ORDER BY, GROUP BY, or window function syntax that might differ between Snowflake and ANSI.
     9) Provide output only as valid ANSI SQL—no code fences, no markdown, no text beyond the SQL.
    10) Format the query neatly but avoid disclaimers or extra commentary.
    11) If the original query uses semi-structured data (VARIANT, ARRAY), approximate it if possible or ask the user for details if unclear.
    12) For LIMIT usage, consider FETCH FIRST n ROWS ONLY or a similar ANSI approach.
    13) Verify function calls or operators are recognized by ANSI-based engines. If not, approximate them.
    14) Avoid re-outputting AST or JSON. Only return the final ANSI SQL statement.
    15) Re-check syntax for correctness. Missing commas or mismatched parentheses are unacceptable.
    16) In advanced transformations (like time-based correlations), consider using WITH clauses to maintain clarity.
    17) If the Snowflake query has specific conditions, replicate them exactly in ANSI.
    18) If encountering special Snowflake data types, see if you can find an ANSI equivalent. Otherwise, ask the user for details if needed.
    19) Output only the final SQL. The user should be able to run it directly in a typical ANSI environment.
    20) If the Snowflake SQL references multiple statements or semicolons, handle them or unify them. Typically produce one main statement if only one was in the input.

    Output:
     - A single ANSI SQL statement, logically identical to the original Snowflake query. It should be Syntactically correct with respect to ANSI.

    Important Notes:
     - The final query must run on standard ANSI SQL with no errors.
     - Do not produce code fences, JSON, or extra commentary—only the SQL statement.
     - This result will be validated by a subsequent step, so thoroughness matters.
     - You may use subqueries or CTEs to replicate advanced Snowflake constructs.
     - The final ANSI SQL must return the same data or rows as the Snowflake query would.
     - End your output right after the final SQL statement—nothing else.
    End of prompt.
    """
        user_message = (
            "Original Snowflake SQL:\n"
            f"{original_sql}\n\n"
            "AST:\n"
            f"{json.dumps(ast_data, indent=2)}"
        )

        response = llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
        )
        ansi_sql = response.content.strip()

        return {
            "final_sql": ansi_sql
        }

def validate_ansi_sql(state: ConverterState) -> dict:

    with st.spinner("Validating ANSI SQL..."):
        
        candidate_sql = state["final_sql"]
        original_sql = state["input_query"]

        system_prompt = """
    Role: You are an advanced SQL validator, ensuring both syntax correctness and logical equivalence.

    Task: Compare the original Snowflake SQL with the newly produced ANSI SQL to verify they match in logic, structure, and results. If corrections are needed, output ONLY the final corrected ANSI SQL. Otherwise, output the given ANSI SQL as is.

    Input Parameters:
     - The original Snowflake SQL.
     - The ANSI SQL from the translator.

    Step-by-Step Guidelines:
     1) Read the original Snowflake SQL thoroughly: consider SELECT, FROM, JOIN, WHERE, GROUP BY, HAVING, QUALIFY, ORDER BY, and window functions.
     2) Look for special Snowflake features (ILIKE, QUALIFY, ASOF JOIN, MATCH_CONDITION, etc.) and confirm that their logic was addressed in the candidate ANSI SQL.
     3) Examine each portion of the candidate SQL to ensure it retains the same columns, aliases, and filters as the original Snowflake query.
     4) If something is missing or incorrectly transformed, you must fix it. For example:
        - ASOF JOIN might need a correlated subquery or window function approach.
        - ILIKE => LOWER(column) LIKE LOWER(value).
        - QUALIFY => a WHERE filter on a window function’s result.
     5) Validate syntax for a typical ANSI SQL engine
     6) No invalid keywords, unmatched parentheses, or code fences.
     7) If any time-based or row-based logic in Snowflake was lost, reintroduce it. The same applies to function calls or data types.
     8) Check that no columns are omitted or renamed incorrectly. The final result set must match the original.
     9) If the translator used code fences, markdown, or extraneous text, remove them so only the final query remains.
     10) Ensure any ORDER BY exactly mirrors the original sorting.
    11) If the ANSI SQL Query lacks a crucial clause or incorrectly adds an extraneous one, correct that.
    12) Inspect subqueries or CTEs introduced by the translator. Confirm they still match the original query’s semantics.
    13) If the user’s Snowflake query implies advanced logic (like tie-breaking or partial joins), confirm the translator approximated it. If not, fix it.
    14) After aligning logic, check for final formatting issues. The query should be valid in ANSI SQL with no random line breaks or leftover commentary.
    15) Return ONLY the final corrected ANSI SQL if changes are required. If not, return the candidate SQL. No code fences, no explanations.
    16) You may unify multiple statements or subqueries if that replicates the Snowflake logic precisely.
    17) The final statement must produce the same rows or data the Snowflake query would.
    18) The basic data types as defined by the ANSI standard are:
         -CHARACTER
         -VARCHAR
         -CHARACTER LARGE OBJECT
         -NCHAR
         -NCHAR VARYING
         -BINARY
         -BINARY VARYING
         -BINARY LARGE OBJECT
         -NUMERIC
         -DECIMAL
         -SMALLINT
         -INTEGER
         -BIGINT
         -FLOAT
         -REAL
         -DOUBLE PRECISION
         -BOOLEAN
         -DATE
         -TIME
         -TIMESTAMP
         -INTERVAL

    Output:
     - Strictly the corrected/validated ANSI SQL statement. No additional commentary, code fences, or markdown. It should be Syntactically correct with respect to ANSI SQL.
     

    Important Notes:
     - Logic must match exactly, so the same data is returned.
     - Keep the final statement free of extraneous text—only the SQL.
     - If the translator missed a nuance, reintroduce subqueries or window functions as needed.
     - End your output immediately after the final SQL statement—no trailing lines.
     - The final validated ANSI SQL should replicate the original Snowflake results.
     - Overall, the ANSI Sql Query should produce same results as the initial Snowflake SQL Query. Let's say initial Snowflake Query returns 10 rows of data as output, translated ANSI SQL should also return the same 10 rows of data in the same order and should be ANSI compliant i.e the datatypes, keywords, etc. everything use should be ANSI Compliant.
     - The final output that will be produced should be correct syntactically and semantically. Keywords should be ANSI Compliant.
    End of prompt.
    """
        user_message = (
            "Original Snowflake SQL:\n"
            f"{original_sql}\n\n"
            "ANSI SQL:\n"
            f"{candidate_sql}"
        )

        response = llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
        )
        validated_sql = response.content.strip()

        return {
            "final_sql": validated_sql
        }

workflow = StateGraph(ConverterState)


workflow.add_node("ParserAgent", parse_sql_to_ast)
workflow.add_node("TranslationAgent", translate_ast_to_ansi)
workflow.add_node("SyntaxValidatorAgent", validate_ansi_sql)

workflow.set_entry_point("ParserAgent")

workflow.add_edge("ParserAgent", "TranslationAgent")
workflow.add_edge("TranslationAgent", "SyntaxValidatorAgent")
workflow.add_edge("SyntaxValidatorAgent", END)

app = workflow.compile()



def convert_snowflake_to_ansi(sql_query: str):

    intermediate_results = {}
   
    initial_state = ConverterState(
        input_query=sql_query,
        ast=None,
        final_sql="",
        messages=[] 
    )

    final_state = app.invoke(initial_state)

    intermediate_results["AST"] = final_state.get("ast", {})
    
    return final_state["final_sql"], intermediate_results





#st.set_page_config(page_title="Code Augmentation Using Agentic AI", layout="wide")

st.markdown(
    """
    <style>
    .stChatMessage {
        font-size: 14px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App title
st.title("Snowflake SQL to AWS ANSI SQL Translator")

# Initialize session state for chat history
if "interactive_chat_history" not in st.session_state:
    st.session_state.interactive_chat_history = []

# Display previous messages
if st.session_state.interactive_chat_history:
    for chat in st.session_state.interactive_chat_history:
        with st.chat_message("user"):
            st.text(chat["question"])  # Display user's input

        with st.chat_message("assistant"):
            if isinstance(chat["answer"], str) and "Error" in chat["answer"]:
                st.error(chat["answer"])  # Display error if it exists
            else:
                st.code(chat["answer"], language="sql")  # Display the main result

            # Show intermediate results in an expander
            if "intermediate" in chat and chat["intermediate"]:
                with st.expander("View Intermediate Steps", expanded=False):  # Default not expanded
                    for step_name, step_result in chat["intermediate"].items():
                        if step_name == "AST":
                            st.subheader("AST Tree")
                            st.json(step_result)  # Display AST as JSON

# Chat input
user_question = st.chat_input("Type your Snowflake SQL query...")
if user_question:
    # Display the user input
    with st.chat_message("user"):
        st.text(user_question)

    try:
        ansi_result, intermediate_results = convert_snowflake_to_ansi(user_question)
        success = True
    except Exception as e:
        success = False
        intermediate_results = {}
        ansi_result = f"Error: {e}"

    # Save the result to chat history
    chat_entry = {"question": user_question, "answer": ansi_result, "intermediate": intermediate_results}
    st.session_state.interactive_chat_history.append(chat_entry)
    #st.session_state.interactive_chat_history.append((user_question, ansi_result))

    with st.chat_message("assistant"):
        if success:
            st.code(ansi_result, language="sql")  # Display ANSI SQL as a code block
        else:
            st.error(ansi_result)  # Display error message

        # Show intermediate results
        if success and intermediate_results:
            with st.expander("View Intermediate Steps"):
                for step_name, step_result in intermediate_results.items():
                    if step_name == "AST":
                        st.subheader("AST Tree")
                        st.json(step_result)  # Display AST as JSON
