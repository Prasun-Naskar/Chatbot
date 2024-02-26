import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
#nltk.download('punkt')


intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
        "responses": ["Hi there", "Hello", "Hey", "I'm fine, thank you", "Nothing much"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": ["Goodbye", "See you later", "Take care"]
    },
    {
        "tag": "doing",
        "patterns": ["what are you doing ", "how's life going", "what you do", "right now"],
        "responses": ["Coding as usual", "Sleeping like dead", "Dancing in space"]
    },
    {
        "tag": "name",
        "patterns": ["what's your name ", "whom to i talking", "tell me about you"],
        "responses": ["I am PRAboT created by Prasun","A chatbot for conversing with you"]
    },
    {
        "tag": "ambition",
        "patterns": ["what you want to be ", "next step in life", "what are your ambitions","aim in life"],
        "responses": ["I am a proficient techie who is passionate about data science","A position and responsibilities of a data scientist thrives and excites me"]
    },
    {
        "tag": "travel",
        "patterns": ["where have you been travelled ", "where could I travel", "share some travelling tips"],
        "responses": ["I have travelled across 30 places all over India",
                      "Depends on you're a mountain or beach person, Best mountains I got in Uttarakhand and best beaches are in vizag",
                      "Travelling means exploring new people and culture, always try to be generous"]

    },
    {
        "tag": "hobbies",
        "patterns": ["what you like to do ", "what are your interests", "exciting things for you"],
        "responses": ["I like to travel a lot ","I am a pretty good cricket player,loves to play it ",
                      "Also I try to learn new technologies for problem solving"]
    },
    {
        "tag": "data science",
        "patterns": ["what is data science ", "explain data science to me", "how can i learn data science", "opportunities of data science "],
        "responses": ["Data Science is a vast concept, you can start with databases",
                      "In today's era every business is using data science for gaining insights about their businesses ",
                      "this provides opportunity to predict and forecast every possible things in our surroundings"]
    },
    {
        "tag": "birthday",
        "patterns": ["its my birthday ", "i born today", "yesterday i had celebrated", "just getting out"],
        "responses": ["Happy birthday wish you lots of love","Its your day enjoy buddy","Happy Birthday and don't forget about party"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome", "No problem", "Glad I could help","Lots of love"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "What are you", "What is your purpose"],
        "responses": ["I am a chatbot", "My purpose is to assist you", "I can answer questions and provide assistance"]
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?", "How can I assist you?"]
    },
    {
        "tag": "age",
        "patterns": ["How old are you", "What's your age"],
        "responses": ["MY creator's age is 24."," I'm  PRAboT and I don't have any age .",
                      "I was just born in the digital world.",
                      "Age is just a number for me."]
    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like", "How's the weather today"],
        "responses": ["I'm sorry, I cannot provide real-time weather information.", "You can check the weather on a weather app or website.",
                      "You can look outside to know about it"]
    },
    {
        "tag": "budget",
        "patterns": ["How can I make a budget", "What's a good budgeting strategy", "How do I create a budget"],
        "responses": ["To make a budget, start by tracking your income and expenses. "
                      "Then, allocate your income towards essential expenses like rent, food, and bills."
                      " Next, allocate some of your income towards savings and debt repayment. Finally, "
                      "allocate the remainder of your income towards discretionary expenses like entertainment and hobbies.",
                      "A good budgeting strategy is to use the 50/30/20 rule. This means allocating 50% of your income towards essential expenses,"
                      " 30% towards discretionary expenses, and 20% towards savings and debt repayment.",
                      "To create a budget, start by setting financial goals for yourself. Then, track your income and expenses"
                      " for a few months to get a sense of where your money is going. Next, create a budget by allocating your income towards essential expenses,"
                      " savings and debt repayment, and discretionary expenses."]
    },
    {
        "tag": "credit_score",
        "patterns": ["What is a credit score", "How do I check my credit score", "How can I improve my credit score"],
        "responses": ["A credit score is a number that represents your creditworthiness. "
                      "It is based on your credit history and is used by lenders to determine whether or not to lend you money. "
                      "The higher your credit score, the more likely you are to be approved for credit.",
                      "You can check your credit score for free on several websites such as Credit Karma and Credit Sesame."]
    }
]


vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for i in intents:
    for p in i['patterns']:
        tags.append(i['tag'])
        patterns.append(p)

# training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response



counter = 0

def main():
    global counter
    st.markdown("<h1 style='text-align: center; color: grey;'>----------PRAboT----------</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: red;'>Welcome, Please type a message and press Enter to start conversation. </h2>", unsafe_allow_html=True)
    counter += 1
    user_input = st.text_input("You:", key=f"user_input_{counter}")

    if user_input:
        response = chatbot(user_input)
        st.text_area("PRAbot:", value=response, height=100, max_chars=None, key=f"chatbot_response_{counter}")

        if response.lower() in ['goodbye', 'bye']:
            st.write("Thank you for chatting with me. Have a great day!")
            st.stop()

if __name__ == '__main__':
    main()