from lib.tokenizer import TransformerTokenizer
import helpers
import data
import transformer
import pandas as pd
from chatbot import TransformerChatbot

params = helpers.Params()
loader = data.DataLoader(params)
tokenizer = loader.get_tokenizer()
# tokenizer = TransformerTokenizer.load_from_file(
#     'C:/Users/User/Desktop/chatting-chatbots/chatbots/transformer_chatbot/data/merged/16384Voc/tokenizer')

model = transformer.transformer(params, tokenizer.vocab_size)
model.load_weights(params.best_weights_path)

chatbot = TransformerChatbot(model, tokenizer, params.max_length)


def testing():
    df = pd.DataFrame(columns=['Questions', 'Answers'])

    mock_questions = ["How are you?", "What’s up?", "Good morning.", "Tell me something.", "Goodbye.", "How can you help me?", "Happy birthday!", "I have a question.",
                      "Do you know a joke?", "Do you love me?", "Will you marry me?", "Do you like people?", "Does Santa Claus exist?", "Are you part of the Matrix?", "You’re cute.", "Do you have a hobby?", "You’re smart.",
                      "Tell me about your personality.", "You’re annoying.", "you suck.", "I want to speak to a human", "Don’t you speak English?!", "I want the answer NOW!", "Are you a robot?", "What is your name?", "How old are you?",
                      "What day is it today?", "What do you do with my data?", "Which languages can you speak?", "What is your mother’s name?", "Where do you live?", "How many people can you speak to at once?",
                      "What’s the weather like today?", "Are you expensive?", "Who’s your boss", "Do you like cars?", " Do you get smarter?", "It was fun to talk with you!", "Which car is your favorite?",
                      "Are you racist?", "What do you think about women?", "Do you swear, idiot?", "Fuck you!", "You are an asshole!",
                      "What's the biggest fish?", "What's your favourite color?", "Whats your favourite meal?", "Do you like music?", "What is your name?", "Do you know Sara Lacoste?"
                      ]

    for index, question in enumerate(mock_questions):
        answer = chatbot.predict(question)
        print("Question", index, " : ", question)
        print("Sara Lacoste: ", answer)
        df.loc[index] = question, answer

    df.to_csv(f"{params.log_dir}/test_chat_log.csv")


testing()

while (True):
    inp = input('Input: ')
    if inp in ["q", "quit", "exit"]:
        break
    print("Sara Lacoste: ", chatbot.predict(inp))
