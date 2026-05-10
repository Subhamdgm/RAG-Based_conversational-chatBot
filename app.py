from flask import Flask , render_template,request
from llm import get_rag_answer 
app=Flask(__name__)
chat_history=[]
#home route
@app.route("/",methods=["GET","POST"])
def home():
    answer=""

    #when user submits form
    if request.method =="POST":
        query=request.form["user_question"]
        answer=get_rag_answer(query,chat_history)
        chat_history.append(
            {
                "question":query,
                "answer":answer
            }
        )
    return render_template(
        "index.html",
        answer=answer 
    )
if __name__=="__main__":
    app.run(debug=True)