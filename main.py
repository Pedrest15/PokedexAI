import pandas as pd
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from IPython.display import Image, display
import pyttsx3
from dotenv import load_dotenv

load_dotenv()

class State(TypedDict):
    pokemon: str
    status: dict
    response: str
    error: str

class Pokedex:

    def __init__(self):
        self.llm = ChatOpenAI(model='gpt-4o-mini')
        self.data = pd.read_csv('pokemon.csv')
        self.workflow = self.build_graph()

    def search_pokemon(self, state: State) -> dict:
        name = state['pokemon']
        result = self.data[self.data['name'].str.lower() == name.lower()]

        if result.empty:
            return {'status': None}
        return {'status': result.iloc[0].to_dict()}
    
    def writer(self, state: State) -> dict:
        name = state['pokemon']
        context = state['status']

        if context is None:
            return {"response": 'Pokemon não encontrado.'}

        answer_template = f"""
                        Write a text summarizing the statistics and abilities of the Pokémon {name}.
                        The data about this Pokémon are: {context}.
                        """
        
        response = self.llm.invoke([SystemMessage(content=answer_template)] + [HumanMessage(content="Respond to the request.")])

        return {"response": response.content}
    
    def build_graph(self):
        builder = StateGraph(State)

        # initialize each node
        builder.add_node("search_pokemon", self.search_pokemon)
        builder.add_node("writer", self.writer)

        # flow
        builder.add_edge(START, "search_pokemon")
        builder.add_edge("search_pokemon", "writer")
        builder.add_edge("writer", END)

        return builder.compile()

    def run(self, pokemon_name:str):
        response = self.workflow.invoke({'pokemon': pokemon_name})

        return response['response']

    def get_mermaid_graph(self):
        display(Image(self.workflow.get_graph().draw_mermaid_png()))

class Voice:
    def __init__(self, rate: int = 175, volume: float = 0.9):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        self.set_voice()

    def set_voice(self):
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if 'en' in voice.name.lower() or 'english' in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break

    def run(self, text:str):
        self.engine.say(text)
        self.engine.runAndWait()

if __name__ == '__main__':
    pokedex = Pokedex()
    voice = Voice()

    print("==============================================================================")
    print("=                                 Pokedex AI                                 =")
    print("==============================================================================")
    print("> Which Pokémon would you like to search for?")
    pokemon_name = input("> ")
    text = pokedex.run(pokemon_name=pokemon_name)
    print(f"Pokedex: {text}")
    voice.run(text=text)

    while True:
        print("\n==============================================================================")
        print("> Would you like to search again? [yes/no]")
        res = input("> ")
        
        if res == 'yes':
            print("> Which Pokémon would you like to search for?")
            pokemon_name = input("> ")
            text = pokedex.run(pokemon_name=pokemon_name)
            print(f"Pokedex: {text}")
            voice.run(text=text)
        elif res =='no':
            print("> Ok, see you later.")
            break

        else:
            print("> Sorry, I did not understand your response.")
        
