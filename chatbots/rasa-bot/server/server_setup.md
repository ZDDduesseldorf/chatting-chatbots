###### Step 1:

ssh root@162.55.50.102   
(replace with your useraccount and server-ip for reproducing)

###### Step 2:

tmux

###### Step 3:

conda activate rasabot;cd ~/repos/rasa-bot/rasa; rasa run actions

###### Step 4:

ctrl + b  c (new tmux window)

###### Step 5:

conda activate rasabot; cd ~/repos/rasa-bot/rasa; rasa run --enable-api --cors "*"

###### Step 6:

ctrl + b  c (new tmux window)

###### Step 6:

ngrok http 5005

###### Step 7:

POST Request: https://d8d0-2a01-4f8-1c1e-a475-00-1.eu.ngrok.io/webhooks/rest/webhook

- with:

```json
{
  "sender": "user_name",
  "message": "Hi, there!"
}
```



****

###### Step 8: (Optional: for keeping active and leaving server enviroment)
- ctrl + b d (tmux detach from session)
- exit (close ssh session)
