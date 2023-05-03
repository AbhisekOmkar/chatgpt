import os
import openai
openai.api_key = os.getenv("OPENAI_KEY") 
messages = [
    {"role": "system", "content": "You are a kind helpful assistant."},
]

messages = []
system_msg = input("Bot: What type of chatbot would you like to create?\nUser: ")
messages.append({"role": "system", "content": system_msg})
if system_msg == 'faq' or system_msg == 'qna':
    message = input("Bot: What would you like to name the bot?\nUser: ")
    messages.append({"role": "user", "content": message})
    message = input("Bot: What data you want to extract\nUser: ")
    messages.append({"role": "user", "content": message})
    messages.append({"role": "system", "content": f'create a follow up questions on ${message} with details stored in excel to automate operation on the columns using excel package'})
else:
    messages.append({"role": "assistant", "content": "Sorry I can create faq and qna bots only. You can ask any further questions or write quit"})
while input != "quit()":
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages)
    reply = response["choices"][0]["message"]["content"]
    message = input(f"Bot: {reply}\nUser: ")
    messages.append({"role": "assistant", "content": reply})
