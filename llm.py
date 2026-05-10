from groq import Groq
import os 
from dotenv import load_dotenv
from semantic_search import top_k_similar


load_dotenv()
client=Groq(
    api_key=os.getenv("GROQ_API_KEY")
    )
def get_rag_answer(query,chat_history):

    enhanced_query=query
    if len(chat_history)>0:
        previous_question =chat_history[-1]["question"]
        previous_answer=chat_history[-1]["answer"]
        enhanced_query=f""" 
        Previous Question:{previous_question}
        Previous answer:{previous_answer}
        current Question:
        {query}
        """
    

    results = top_k_similar(enhanced_query)
    #build context
    combined_context=""
    for result in results:
        combined_context += result["chunk"]
        combined_context +="\n\n"
    print()
    print("combined_context")

    prompt=f"""
        Example:

        User: What is machine learning?
        Assistant: Machine learning is a subset of AI.

        User: Describe it
        Assistant: Machine learning enables systems to learn from data.


        Now answer the real question.

        Context:
        {combined_context}

        Question:
        {query}
            """

    #send to groq 
    response = client.chat.completions.create(
        model ="llama-3.1-8b-instant",
        messages=[
            {
            "role": "system",
            "content":
            """you are a helpful AI assistant.Answer only from the provided
            context in 50 words. """
            },
            {
                "role":"user",
                "content":prompt 
            }
        ]
    )
    answer = response.choices[0].message.content 
    return answer
