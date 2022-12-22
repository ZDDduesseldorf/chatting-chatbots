import helpers
import data
import transformer
from chatbot import TransformerChatbot

params = helpers.Params()
loader = data.DataLoader(params)
tokenizer = loader.get_tokenizer()

model = transformer.transformer(params, tokenizer.vocab_size)
model.load_weights(params.weights_path)

chatbot = TransformerChatbot(model, tokenizer, params.max_length)

while (True):
    inp = input('>> ')
    if inp in ["q", "quit", "exit"]:
        break
    print(chatbot.predict(inp))
