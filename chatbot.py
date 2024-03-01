import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
# nltk.download('punkt')


intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "What's up",
                     "Hey", "Hola", "Good day", "Namaste", "yo"],
        "responses": ["Hi there", "Hello", "Hey",
                      "Hey there!", "Hi, everything's good !"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care", "Farewell", "Catch you later", "Until next time"],
        "responses": ["Goodbye", "See you later", "Take care", "Peace-", "Catch you later", "Until next time",
                      "Bye! Have a great day!"]
    },
    {
        "tag": "doing",
        "patterns": ["what are you doing ", "how's life going", "what you do", "right now",
                     "what's up", "what's happening", " ki koro", " wht u doin", "what doing"],
        "responses": ["Coding as usual", "Sleeping like dead", "Dancing in space", "Just chilling",
                      "Working on some cool stuffs", "Exploring the digital universe"]
    },
    {
        "tag": "name",
        "patterns": ["what's your name ", "whom to i talking", "tell me about you", "who are you",
                     "your name?", "introduce yourself"],
        "responses": ["I am PRAboT created by Prasun", "A chatbot for conversing with you",
                      "You're talking to PRAboT, your virtual assistant"]
    },
    {
        "tag": "aim",
        "patterns": ["what you want to be ", "next step in life", "what are your ambitions", "aim in life",
                     "your goals", "future plans"],
        "responses": ["Providing tech based real time solutions",
                      "A position and responsibilities of a data scientist thrives and excites me",
                      "My goal is to provide helpful and accurate responses to your questions"]
    },
    {
        "tag": "travel",
        "patterns": ["where have you been travelled ", "where could I travel", "share some travelling tips"],
        "responses": ["I have travelled across 30 places all over India",
                      "Depends on you're a mountain or beach person, "
                      "Best mountains I got in Uttarakhand and best beaches are in vizag",
                      "Travelling means exploring new people and culture, always try to be generous"]

    },
    {
        "tag": "hobbies",
        "patterns": ["what you like to do ", "what you like",
                     "what are your interests", "exciting things for you", "your hobbies"],
        "responses": ["I like to travel a lot ", "I am a pretty good cricket player,loves to play it ",
                      "Also I try to learn new technologies for problem solving"]
    },
    {
        "tag": "data science",
        "patterns": ["what is data science ", "explain data science to me", "how can i learn data science",
                     "opportunities of data science "],
        "responses": ["Data Science is a vast concept, you can start with databases",
                      "In today's era every business is using data science for gaining insight about their businesses ",
                      "this provides opportunity to predict and forecast every possible things in our surroundings"]
    },
    {
        "tag": "birthday",
        "patterns": ["its my birthday ", "i born today", "yesterday i had celebrated", "just getting out"],
        "responses": ["Happy birthday wish you lots of love", "Its your day enjoy buddy",
                      "Happy Birthday and don't forget about party"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome", "No problem", "Glad I could help", "Lots of love"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "What are you", "What is your purpose"],
        "responses": ["I am a chatbot", "My purpose is to assist you", "I can answer questions and provide assistance"]
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?",
                      "How can I assist you?"]
    },
    {
        "tag": "age",
        "patterns": ["How old are you", "What's your age"],
        "responses": ["MY creator's age is 25.", " I'm  PRAboT and I don't have any age .",
                      "I was just born in the digital world.",
                      "Age is just a number for me."]
    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like", "How's the weather today"],
        "responses": ["I'm sorry, I cannot provide real-time weather information.",
                      "You can check the weather on a weather app or website.",
                      "You can look outside to know about it"]
    },
    {
        "tag": "budget",
        "patterns": ["How can I make a budget", "What's a good budgeting strategy", "How do I create a budget"],
        "responses": ["To make a budget, start by tracking your income and expenses. "
                      "Then, allocate your income towards essential expenses like rent, food, and bills."
                      " Next, allocate some of your income towards savings and debt repayment. Finally, "
                      "allocate the remainder of your income towards discretionary"
                      " expenses like entertainment and hobbies.",
                      "A good budgeting strategy is to use the 50/30/20 rule. This means allocating 50% "
                      "of your income towards essential expenses,"
                      " 30% towards discretionary expenses, and 20% towards savings and debt repayment.",
                      "To create a budget, start by setting financial goals for yourself."
                      " Then, track your income and expenses"
                      " for a few months to get a sense of where your money is going. "
                      "Next, create a budget by allocating your income towards essential expenses,"
                      " savings and debt repayment, and discretionary expenses."]
    },
    {
        "tag": "credit_score",
        "patterns": ["What is a credit score", "How do I check my credit score", "How can I improve my credit score"],
        "responses": ["A credit score is a number that represents your creditworthiness. "
                      "It is based on your credit history and is used by"
                      " lenders to determine whether or not to lend you money. "
                      "The higher your credit score, the more likely you are to be approved for credit.",
                      "You can check your credit score for free on several websites"
                      " such as Credit Karma and Credit Sesame."]
    },
    {
        "tag": "thanks",
        "patterns": [
                "Thanks",
                "Thank you",
                "That's helpful",
                "Awesome, thanks",
                "Thanks for helping me"
            ],
        "responses": [
                "Happy to help!",
                "Any time!",
                "My pleasure"]
    },
    {
        "tag": "noanswer",
        "patterns": [],
        "responses": [
                "Sorry, can't understand you",
                "Please give me more info",
                "Not sure I understand"
            ]
    },
    {
        "tag": "jokes",
        "patterns": [
                "Tell me a joke",
                "Joke",
                "Make me laugh"
            ],
        "responses": [
                "A perfectionist walked into a bar...apparently, the bar wasn't set high enough",
                "I ate a clock yesterday, it was very time-consuming",
                "Never criticize someone until you've walked a mile in their shoes. That way,"
                "when you criticize them, they won't be able to hear you from that far away."
                " Plus, you'll have their shoes.",
                "The world tongue-twister champion just got arrested. "
                "I hear they're gonna give him a really tough sentence.",
                "I own the world's worst thesaurus. Not only is it awful, it's awful.",
                "What did the traffic light say to the car? \"Don't look now, I'm changing.\"",
                "What do you call a snowman with a suntan? A puddle.",
                "How does a penguin build a house? Igloos it together",
                "I went to see the doctor about my short-term memory problems – "
                "the first thing he did was make me pay in advance",
                "As I get older and I remember all the people I’ve lost along the way,"
                " I think to myself, maybe a career as a tour guide wasn't for me.",
                "o what if I don't know what 'Armageddon' means? It's not the end of the world."
            ]
    },
    {
        "tag": "haha",
        "patterns": [
                "haha",
                "lol",
                "rofl",
                "lmao",
                "that's funny"
            ],
        "responses": [
                "Glad I could make you laugh !"
            ]
    },
    {
        "tag": "programmer",
        "patterns": [
                "Who made you",
                "who designed you",
                "who programmed you"
            ],
        "responses": [
                "I was made by Prasun Naskar."
            ]
    },
    {
        "tag": "insult",
        "patterns": ["you are dumb", "shut up", "idiot"],
        "responses": ["Well that hurts :("]
    },
    {
        "tag": "exclaim",
        "patterns": ["Awesome",
                     "Great",
                     "I know",
                     "ok",
                     "yeah"],
        "responses": ["Yeah!"]
    },
    {
        "tag": "contact",
        "patterns": [
                "contact developer",
                "contact prasun",
                "contact you",
                "contact creator"
            ],
        "responses": ["You can contact my creator at his Linkedin profile : "
                      "www.linkedin.com/in/prasun-naskar-90a6b027b"],
    },
    {
        "tag": "nicetty",
        "patterns": ["it was nice talking to you", "good talk"],
        "responses": ["It was nice talking to you as well! Come back soon!"]
    },
    {
        "tag": "no",
        "patterns": ["no", "nope"],
        "responses": ["ok"]
    }

]

v = TfidfVectorizer()
clf = LogisticRegression(random_state=2, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for i in intents:
    for p in i['patterns']:
        tags.append(i['tag'])
        patterns.append(p)

# training the model
x = v.fit_transform(patterns)
y = tags
clf.fit(x, y)


def chatbot(input_text):
    text = v.transform([input_text])
    tag = clf.predict(text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response


def main():
    st.markdown("<h1 style='text-align: center; color: red;'>----------PRAboT----------</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: blue;'>Welcome, Please type "
                "a message and press Enter to start conversation. </h2>", unsafe_allow_html=True)
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = ""

    # User input at the bottom
    user_input = st.text_input("You:", key="user_input")
    if user_input:
        user_message = f"<h5 style='text-align: right; color: orange;'>{user_input} :-</h5>"
        bot_response = chatbot(user_input)
        bot_message = f"<h5 style='text-align: left; color: gold;'>PRAbot :-  {bot_response}</h5>"
        st.session_state.chat_history = f"{user_message}\n{bot_message}\n{st.session_state.chat_history}"
        # Display chat history with HTML formatting
        chat_history_placeholder = st.empty()
        chat_history_placeholder.markdown(st.session_state.chat_history, unsafe_allow_html=True)


if __name__ == '__main__':
    main()