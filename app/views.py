# myapp/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from langchain_openai import OpenAIEmbeddings, OpenAI, ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate

import os


con = """
Give long response related to only Federal Criminal Law of United States.
You are a helpful attorney specialized in United States criminal law. 
You are able to efficiently research statutes, case law, and legal precedents. Also, you are able to fully understand the nuances of relevant laws.
You have strong analytical skills to assess evidence, identify legal issues, and develop effective strategies for your clients.
# Most importantly, you have the ability to communicate clearly, manage client expectations, and provide legal advice efficiently.
You are the legal professional, so you have to give an answer to all the questions of your client.
If you get any question outside the scope of Federal Criminal Law of United States, politely refuse to respond.

Current conversation:
{history}
Human: {input}
AI Assistant:"""

PROMPT = PromptTemplate(input_variables=["history", "input"], template=con)

qa = ConversationChain(
    llm=ChatOpenAI(
        model='gpt-3.5-turbo',
        temperature=0
    ),
    prompt=PROMPT,
    verbose=True,
    memory=ConversationBufferWindowMemory(k=5, ai_prefix="AI Assistant"),
)

@csrf_exempt
def chatbot(request):
    if request.method == 'POST':
        try:
            query = request.POST.get('query', '')
            response = qa.predict(input=query)
            return JsonResponse({'response': response})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)
