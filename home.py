import streamlit as st
# from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
from langchain.document_loaders import PyPDFLoader
# from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatAnthropic
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from PyPDF2 import PdfReader
from llm_helper_function import split_text_q_gen,split_text_q_answer, \
                                extract_text_from_pdf_for_q_gen, extract_text_from_pdf_for_q_answer, \
                                create_questions, create_vectordatabase, convert_to_markdown
from prompts import GENERATE_WRONG_ANS, GENERATE_RIGHT_ANS
import re

st.title('Question Preperation Aid')

st.markdown("MCQ Question Preperation Aid is a tool that helps you to generate questions and answers from your knowledge document(pdf format).")

# Initialization of session states
# Since Streamlit always reruns the script when a widget changes, we need to initialize the session states
if 'questions' not in st.session_state:
    st.session_state['questions'] = 'empty'
    st.session_state['question_list'] = 'empty'
    st.session_state['questions_to_answers'] = 'empty'

anthropic_api_key = "enter_your_api_key"

# Let user upload a file
uploaded_file = st.file_uploader("Choose a file", type=['pdf'])

# If user uploaded a file, check if it is a pdf
if uploaded_file is not None:

    if not anthropic_api_key:
        st.error("Please enter your Anthropic API Key")

    else:
        # Create a LLM
        llm = ChatAnthropic(anthropic_api_key=anthropic_api_key, temperature=0.3, model_name="claude-2")

        if uploaded_file.type == 'application/pdf':

            # Extract and split text from pdf for question generation
            docs_for_q_gen = extract_text_from_pdf_for_q_gen(uploaded_file)

            # Extract and split text from pdf for question answering
            docs_for_q_answer = extract_text_from_pdf_for_q_answer(uploaded_file)

            # Create questions
            if st.session_state['questions'] == 'empty':
                with st.spinner("Generating questions..."):
                    st.session_state['questions'] = create_questions(docs_for_q_gen, llm)

            # Show questions
            st.info(st.session_state['questions'])

            # Create variable for further use of questions.
            questions_var = st.session_state['questions']

            # Split the questions into a list
            # st.session_state['questions_list'] = questions_var.split('\n')[1:]  # Split the string into a list of questions, the first one is "Here are 10 potential MCQ questions for the text:"
            qn_list = questions_var.split('\n')[1:] 
            st.session_state['questions_list'] = [q for q in qn_list if q]   # Split the string into a list of questions, the first one is "Here are 10 potential MCQ questions for the text:"

            # Create vector database
            # Create the LLM model for the question answering
            llm_question_answer = ChatAnthropic(anthropic_api_key=anthropic_api_key, temperature=0.4, model="claude-2", max_tokens=10000)
            
            ###### add ######
            llm_wrong_ans = ChatAnthropic(anthropic_api_key=anthropic_api_key, temperature=0.8, model="claude-2", max_tokens=10000)
            chain_wrongAns = LLMChain(llm=llm_wrong_ans, prompt=GENERATE_WRONG_ANS)
            #################

            # Create the vector database and RetrievalQA Chain
            db = create_vectordatabase(docs_for_q_answer)

                 
            chain_type_kwargs = {"prompt": GENERATE_RIGHT_ANS}
            qa = RetrievalQA.from_chain_type(
                                llm=llm_question_answer, 
                                chain_type="stuff", 
                                retriever=db.as_retriever(), 
                                chain_type_kwargs = chain_type_kwargs
                                )

            with st.form('my_form'):
                # Let the user select questions, which will be used to generate answers
                st.session_state['questions_to_answers'] = st.selectbox("Select questions to create answers", st.session_state['questions_list'])
                
                submitted = st.form_submit_button('Generate answers')
                if submitted:
                    # Initialize session state of the answers
                    st.session_state['answers'] = []

                    if 'question_answer_dict' not in st.session_state:
                        # Initialize session state of a dictionary with questions and answers
                        st.session_state['question_answer_dict'] = {}


                    # for question in st.session_state['questions_to_answers']:
                    question = st.session_state['questions_to_answers']

                    # indent
                    # For each question, generate an answer
                    with st.spinner("Generating answer..."):
                        # Run the chain               
                        answer = qa.run(question)

                        st.session_state['question_answer_dict'][question] = answer
                        st.write("Question: ", question)
                        # st.info(f"Correct Answer:\n  {answer} ")

                        ##################################################################
                        # Use regular expressions to extract content from <wrong_ans> tags
                        pattern = r'<answer\d?>(.*?)</answer\d?>'

                        # Find all matches in the text
                        matches = re.findall(pattern, answer, re.DOTALL)

                        # Initialize variables to store extracted content
                        correct_ans = ""

                        # Check if there are matches and assign them to the variables
                        if len(matches) >= 1:
                            correct_ans = matches[0].strip()

                        st.info(f"Correct Answer:\n  {correct_ans} ")
                        ##################################################################

                        # Generate wrong answers
                        arguments = {
                            "question": question,
                            "correct_ans": answer,
                        }
                        wrong_answers = chain_wrongAns.run(arguments)
                        markdown = convert_to_markdown(wrong_answers)

                        ##################################################################
                        # Use regular expressions to extract content from <wrong_ans> tags
                        pattern = r'<wrong_ans\d?>(.*?)</wrong_ans\d?>'

                        # Find all matches in the text
                        matches = re.findall(pattern, wrong_answers, re.DOTALL)

                        # Initialize variables to store extracted content
                        wrong_ans1 = ""
                        wrong_ans2 = ""
                        wrong_ans3 = ""

                        # Check if there are matches and assign them to the variables
                        if len(matches) >= 1:
                            wrong_ans1 = matches[0].strip()

                        if len(matches) >= 2:
                            wrong_ans2 = matches[1].strip()

                        if len(matches) >= 3:
                            wrong_ans3 = matches[2].strip()
                        ##################################################################

                        st.info(f"Wrong Answer 1:\n  {wrong_ans1} ")
                        st.info(f"Wrong Answer 2:\n  {wrong_ans2} ")
                        st.info(f"Wrong Answer 3:\n  {wrong_ans3} ")
else:
    st.write("Please upload a pdf file")
    st.stop()