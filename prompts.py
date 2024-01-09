from langchain.prompts import PromptTemplate

prompt_template_questions = """
Human: 
Your goal is to prepare 10 questions for MCQ to test students in an exam. Do not prepare the answers.
Ask questions about text between <text> and </text>:
<text>
{text}
</text>

Each of the 10 questions should only consist of one specific question.
Make sure not to lose any important information.\n

Assistant:
"""
PROMPT_QUESTIONS = PromptTemplate(template=prompt_template_questions, input_variables=["text"])


refine_template_questions = ("""
Human: 
Your goal is to prepare 10 questions for MCQ to test students in an exam. 
We have received 10 questions to a certain extent between <existing_answer> and </existing_answer>: 
<existing_answer>
{existing_answer}
</existing_answer>.
We have the option to refine the existing questions (only if necessary) with some more context between <text> and </existing_answer>
"------------\n"
<text>
"{text}\n"
</text>
"------------\n"
  
Given the new context, refine the original questions in English.
Each question should only contain one single question.
Do not include answer to MCQ in your response.
If the context is not helpful, please provide the original questions. Make sure to be detailed in your questions.
Respond only with 10 questions.\n

Assistant:
"""
)
REFINE_PROMPT_QUESTIONS = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refine_template_questions,
)


# Prepare prompt to limit correct answer to be within 3 sentences, used by RetrievalQA
## Reference: https://python.langchain.com/docs/modules/chains/popular/vector_db_qa
GENERATE_RIGHT_ANS = PromptTemplate.from_template("""
Human: 
Use the following pieces of context to answer the question inside <question></question>. 
<context>
{context}
</context>

Question: 
<question>
{question}
</question>

When you reply, first find exact quotes in the context relevant to the user's question and 
write them down word for word inside <thinking></thinking> XML tags.  
This is a space for you to write down relevant content and will not be shown to the user. 
Your thoughts about the context is not included as response to the user.                                                   
Once you are done extracting relevant quotes, answer the question.  
Think step by step to answer the question.
Put your answer to the user inside <answer></answer> XML tags.

Do not use bullet points, or numbered list in your answer.                                      
Your answer needs to be less than 3 sentences.

Assistant: 
""")


GENERATE_WRONG_ANS = PromptTemplate.from_template("""
Human: 
Your goal is to prepare 3 wrong answers for question provided, for use to test students in an exam. 

The question is defined in text delimited by triple hash.
###
<question>
{question}
</question>
###

The correct answer for the question is defined in text delimited by triple backticks.
```
<correct_ans>
{correct_ans}
<\correct_ans>
```

Each wrong answer have same format as compared to correct answer
Each wrong answer have same number of sentences as compared to correct answer
Each wrong answer miss some facts from correct answer.
Do not apply negation or antonyms of words from the correct answer.

Do not use list, or bullet points in your response.
Generate 3 different wrong answers using the correct answer.
                                                  
Prepare 3 wrong answers in the following format:
<wrong_ans1>...wrong answer 1...</wrong_ans1>
<wrong_ans2>...wrong answer 2...</wrong_ans2>
<wrong_ans3>...wrong answer 3...</wrong_ans3>

Assistant:
""")