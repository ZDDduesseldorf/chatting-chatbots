import requests

def interact():
    public_url = "https://d8d0-2a01-4f8-1c1e-a475-00-1.eu.ngrok.io" #changes after manual vm restart
    rest_suburl = "/webhooks/rest/webhook"#might switch to websocket in the near future after getting the bot integrated into moderation_bot
    url = public_url+rest_suburl
    quit_strings = ["Bye", "bye", "quit", "q", "Quit"]
    quit= False
    while not quit:
        user_input = input('Your Message:')
        if user_input in quit_strings: 
            quit = not quit
        else:
            user_message = {'sender': 'test_user', 'message': user_input }
            response = requests.request("POST", url, json=user_message).json()
            print(response[0]['text'])

# response message for moderator: all other services run inside 3 tmux windows on vm
def response_function(message):
    public_url = "https://d8d0-2a01-4f8-1c1e-a475-00-1.eu.ngrok.io" #changes after manual vm restart
    rest_suburl = "/webhooks/rest/webhook"#might switch to websocket in the near future after getting the bot integrated into moderation_bot
    url = public_url+rest_suburl
    user_message = {'sender': 'moderator', 'message': message}
    response = requests.request("POST", url, json=user_message).json()
    return(response[0]['text'])

# def chat(response_function, bot_name, call_with_conversation=False):
    # asyncio.run(main(response_function, bot_name, call_with_conversation))
    

interact()

