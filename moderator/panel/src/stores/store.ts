import { defineStore } from 'pinia';

export type StoreState = {
  conversation: Message[];
  bots: object[];
  voices: SpeechSynthesisVoice[],
  rankings: Ranking[],
};

export type Message = {
  id: string;
  message: string;
  bot_id: string;
  bot_name: string;
  ranking_number: number;
  topic_score: number;
  similarity_score: number;
  share_score: number;
};

export type Bot = {
  id: string;
  name: string;
  method: string;
  voice: SpeechSynthesisVoice;
};

export type Ranking = {
  message: Message;
  ranked_responses: Message[];
};

export const useStore = defineStore('store', {
  state: () => ({
    conversation: [],
    bots: [],
    voices: [],
    rankings: [],
  } as StoreState),
  getters: {
    /**
     * @returns {string}
     */
    lastMessage(state) {
      return this.conversation[this.conversation.length-1]
    },
  },
  actions: {
    addMessage(message: Message) {
      this.conversation.push(message)
    },
    addBot(bot: Bot) {
      console.log(bot);
      const randomIndex = Math.floor(Math.random() *  this.voices.length);
      bot.voice = this.voices[randomIndex];
      this.voices.splice(randomIndex, 1);
      this.bots.push(bot);
      return bot;
    },
    setVoices(voices: SpeechSynthesisVoice[]) {
      this.voices = voices;
    },
    addMessageRanking(message: Message, response: Message) {
      if (this.rankings.length === 0 || message.id !== this.rankings[this.rankings.length - 1].message.id) {
        this.rankings.push({message: message, ranked_responses: []})
      }
      let rankedResponses = this.rankings[this.rankings.length - 1].ranked_responses;
      rankedResponses.push(response);
      rankedResponses.sort((a: Message, b: Message) => {return a.ranking_number - b.ranking_number});
      rankedResponses.reverse();

    },
  },
})