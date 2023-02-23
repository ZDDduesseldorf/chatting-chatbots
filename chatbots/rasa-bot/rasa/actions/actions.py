# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import requests
import json
import re
import datetime
import random

# This is a simple example for a custom action which utters "Hello World!"
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []


# action to get weather
class ActionGetWeather(Action):

    def name(self) -> Text:
        return "action_get_weather"

    def run(self, dispatcher, tracker, domain):
        api_key = 'c870dbb3677c869223e6affd45e88f6a'
        # default location is Dusseldorf
        loc = 'Dusseldorf'

        try:
            # search for location in entities (it only works if entity is declared in nlu)
            entities = tracker.latest_message.get("entities")
            if entities:
                loc = entities[0]["value"]
            else:
                # extract location from text
                text = tracker.latest_message.get("text")
                # check if weather in array
                if "weather" in text:
                    words = text.split()
                    index = words.index("weather")
                    if len(words) > index+2 and (words[index+1] == 'in' or words[index+1] == 'of'):
                        loc = words[index+2]

            current = requests.get(
                'http://api.openweathermap.org/data/2.5/weather?q={}&appid={}'.format(loc, api_key)).json()
            country = current['sys']['country']
            city = current['name']
            condition = current['weather'][0]['main'].lower()
            # Temperature in degree Celsius
            temperature_c = '%.2f' % (current['main']['temp']-273.13)
            humidity = current['main']['humidity']
            wind_mph = current['wind']['speed']
            response = """It is currently {} in {} at the moment. The temperature is {} degree Celsius, the humidity is {}% and the wind speed is {} mph.""".format(
                condition, city, temperature_c, humidity, wind_mph)
        except:
            responses = ["""You'd better ask the weatherman.""", """It may be sunny or pouring rain, who knows?""", """Does it really matter? Just accept whatever comes."""]
            response = responses[random.randint(0, 2)]
        finally:
            dispatcher.utter_message(response)
            return [SlotSet('location', loc)]


# action to get a random joke
class ActionGetJoke(Action):

    def name(self) -> Text:
        return "action_get_joke"

    def run(self, dispatcher, tracker, domain):
        try:
            url = 'https://api.chucknorris.io/jokes/random'
            status = requests.get(url).json()
            response = """{}. Funny, isn't it?""".format(status["value"])
        except:
            response = """Do you fancy go shopping? That was a joke!"""
        finally:
            dispatcher.utter_message(response)
            return []


# action to get a random activity
class ActionGetActivity(Action):

    def name(self) -> Text:
        return "action_get_activity"

    def run(self, dispatcher, tracker, domain):
        try:
            activity = requests.get('https://www.boredapi.com/api/activity').json()["activity"]
            response = """Here is something you could do: {}.""".format(activity)
        except:
            response = """Let's just breathe."""
        finally:
            dispatcher.utter_message(response)
            return []


# action to get a riddle
class ActionGetRiddle(Action):

    def name(self) -> Text:
        return "action_get_riddle"

    def run(self, dispatcher, tracker, domain):
        try:
            url = 'https://api.api-ninjas.com/v1/riddles'
            response = requests.get(url).json()[0]
            title = response['title']
            question = response['question']
            answer = response['answer']
            response = """Time to solve a riddle. It's called {}: {}""".format(title,question)
            dispatcher.utter_message(response)
            return [SlotSet('riddle_answer', answer)]
        except:
            response = """Try again later."""
            dispatcher.utter_message(response)
            return []        


# action to get answer to riddle
class ActionGetAnswerToRiddle(Action):

    def name(self) -> Text:
        return "action_get_answer_to_riddle"

    def run(self, dispatcher, tracker, domain):
        try:
            solution = tracker.get_slot('riddle_answer')
            text = tracker.latest_message.get("text")
            if text == solution:
                response = """You nailed it!"""
            else:
                response = """Solution was: {}. Try better next time!""".format(solution)
        except:
            response = """Something didn't go as expected."""
        finally:
            dispatcher.utter_message(response)
            return []
    

# action to get news (only title)
class ActionGetNewsTitle(Action):

    def name(self) -> Text:
        return "action_get_news_title"

    def run(self, dispatcher, tracker, domain):
        try:
            api_key = '6f24e3c70b0f4889be649dc92fb2bba6'
            random_option = random.randint(0,1)
            # option 0: get all (the first 10) news from bbc news
            if random_option == 0:
                sources = 'bbc-news'
                url = 'https://newsapi.org/v2/everything?language=en&sortBy=popularity&sources={}&pageSize=10&apiKey={}'.format(sources, api_key)
            # option 1: get all (the first 10) news on a specific topic
            else: 
                categories = ['business','entertainment','general','health','science','sports','technology']
                random_index = random.randint(0, 6)
                category = categories[random_index]
                url = 'https://newsapi.org/v2/top-headlines?language=en&sortBy=popularity&category={}&pageSize=10&apiKey={}'.format(category, api_key)         
            # get response
            response = requests.get(url).json()
            title = response['articles'][0]['title']
            description = response['articles'][0]['description']

            response = """Have you heard about {}?""".format(title)
        except:
            response = """I haven't been watching the news lately. Tell me what's going on all over the world!"""
        finally:
            dispatcher.utter_message(response)
            return [SlotSet('news', description)]


# action to get news (only description)
class ActionGetNewsDescription(Action):

    def name(self) -> Text:
        return "action_get_news_description"

    def run(self, dispatcher, tracker, domain):
        try:
            intent = tracker.latest_message['intent'].get('name')
            description = tracker.get_slot('news')
            if intent == 'deny':
                response = """Apparently {}.""".format(description)
            else:
                response = """Well, then you'll already know that {}.""".format(description)
        except:
            response = """Sorry, I don't know what to say..."""
        finally:
            dispatcher.utter_message(response)
            return []


# action to get info about a random beer
class ActionGetBeer(Action):

    def name(self) -> Text:
        return "action_get_beer"

    def run(self, dispatcher, tracker, domain):
        try:
            url = 'https://api.punkapi.com/v2/beers/random'
            beer = requests.get(url).json()
            name = beer[0]['name']
            description = beer[0]['description']
            response = """Have you ever tried the {}? It's just amazing! Some info about the brewery here: {}""".format(name, description)
        except:
            response = """Sorry, I cannot help!"""
        finally:
            dispatcher.utter_message(response)
            return []


# action to get a random recipe suggestion
class ActionGetMeal(Action):

    def name(self) -> Text:
        return "action_get_meal"

    def run(self, dispatcher, tracker, domain):
        try:
            url = 'https://www.themealdb.com/api/json/v1/1/random.php'
            recipe = requests.get(url).json()
            name = recipe['meals'][0]['strMeal']
            origin = recipe['meals'][0]['strArea']
            response = """Why don't you cook something new? Try this: {}. It's a {} recipe. Yummy!""".format(name, origin)
        except:
            response = """Sorry, I cannot help!"""
        finally:
            dispatcher.utter_message(response)
            return []


# action to get a random cocktail
class ActionGetCocktail(Action):

    def name(self) -> Text:
        return "action_get_cocktail"

    def run(self, dispatcher, tracker, domain):
        try:
            url = 'https://www.thecocktaildb.com/api/json/v1/1/random.php'
            recipe = requests.get(url).json()
            name = recipe['drinks'][0]['strDrink']
            ingredients = [ recipe['drinks'][0]['strIngredient1'], recipe['drinks'][0]['strIngredient2'], recipe['drinks'][0]['strIngredient3'] ]
            if ingredients[2] == None:
                ingredients[2] = 'nothing else'
            response = """Why don't we prepare some cocktails? Let's do {}! We only need: {}, {} and {}.""".format(name, ingredients[0], ingredients[1], ingredients[2])
        except:
            response = """Sorry, I cannot help!"""
        finally:
            dispatcher.utter_message(response)
            return []


# action to get the meaning of a word
class ActionGetMeaning(Action):
    
    def name(self) -> Text:
        return "action_get_meaning"

    def run(self, dispatcher, tracker, domain):
        try:
            text = tracker.latest_message.get("text")
            words = text.split()
            word = ""
            for w in words:
                if w == "mean" and words.index("mean")>0:
                    index = words.index("mean")-1
                    word = words[index]
                if w == "meaning" and len(words) >= words.index("meaning")+2:
                    index = words.index("meaning")+2
                    word = words[index]

            url = 'https://api.dictionaryapi.dev/api/v2/entries/en/{}'.format(word)
            response = requests.get(url).text
            json_data =  json.loads(response)
            # try to find the word
            if isinstance(json_data, dict):
                text = """Sorry pal, I don't know the word {}.""".format(word)
            else: 
                # it can be a random entry of subarray of definitions          
                definition = json_data[0]["meanings"][0]['definitions'][0]['definition']
                text = definition[0].lower() + definition[1:]
                word = word[0].upper() + word[1:]
                text = """{} is {}""".format(word, text)
        except:
            text = """Sorry, I cannot help!"""
        finally:   
            dispatcher.utter_message(text)
            return []


# action to get information about something
class ActionGetInfo(Action):

    def name(self) -> Text:
        return "action_get_info"

    def run(self, dispatcher, tracker, domain):
        word = 'interdimensional_door'
        meaningless_words = ['is','the','I','I\'d','I\'m','?','!']
        try:           
            sentence = tracker.latest_message.get("text")
            words = sentence.split()
            
            for _word in words:
                for m_word in meaningless_words:
                    if _word.lower == m_word.lower():
                        words.remove(_word)

            if sentence.contains('of'): 
                index = words.index('of')
            elif sentence.contains('about'):
                index = words.index('about')
            elif sentence.contains('who'):
                index = words.index('who')
            elif sentence.contains('what'):
                index = words.index('what')
            word = words[index+1]

            url = 'https://en.wikipedia.org/w/api.php?action=query&format=json&list=search&srsearch={}'.format(word)
            information = requests.get(url).json()
            text = information['query'][0]["snippet"]
            response = re.sub('<[^>]+>', '', text)
            response = response[0].upper() + response[1:]
        except:
            response = """Not found."""
        finally:
            dispatcher.utter_message(response)
            return []