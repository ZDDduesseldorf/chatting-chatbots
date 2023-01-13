<script lang="ts">
import Navigation from "./ModeratorSection/Navigation.vue";
import Chat from "./ChatSection/Chat.vue";
import Connections from "./BotSection/Connections.vue";
import Rankings from "./RankingSection/Rankings.vue";
import { useStore } from "@/stores/store";
import { usePusher } from "@/stores/pusher";
export default {
  components: {
    Chat,
    Connections,
    Rankings,
    Navigation,
  },
  data() {
    return {
      utterance: null,
    };
  },
  setup() {
    const pusher = usePusher();
    const store = useStore();
    const synth = window.speechSynthesis;
    return { pusher, store, synth };
  },
  mounted() {
    this.bindBotRegisterd();
    this.bindMessageSent();
    this.bindMessageRanked();
  },
  methods: {
    speak(text: string, voice: SpeechSynthesisVoice = null) {
      this.utterance = new SpeechSynthesisUtterance(text);
      console.log(this.utterance);
      if (voice) {
        this.utterance.voice = voice;
      }
      this.utterance.rate = 0.7;
      this.synth.speak(this.utterance);
      this.utterance.onstart = (event) => {
        console.log(`Utterance started: ${event}`);
      };
      this.utterance.onerror = (event) => {
        console.log(`Utterance error: ${event.error}`);
      };
    },
    bindMessageSent() {
      this.pusher.channel.bind("message-sent", (data: Message) => {
        this.store.addMessage(data);
        const bot = this.store.bots.find((bot) => bot.id === data.bot_id);
        this.speak(data.message, bot ? bot.voice : null);
        this.utterance.onend = (event) => {
          this.pusher.channel.trigger(
            `client-message-done-${data.id}`,
            data.id
          );
        };
        setTimeout(() => {
          document.getElementById("conversation").scrollTo({
            top: document.getElementById("conversation").scrollHeight,
            left: 0,
            behavior: "smooth",
          });

          document.getElementById("rankings").scrollTo({
            top: document.getElementById("rankings").scrollHeight,
            left: 0,
            behavior: "smooth",
          });
        }, 350);
      });
    },
    bindMessageRanked() {
      this.pusher.channel.bind("message-ranked", (data: Message) => {
        this.store.addMessageRanking(
          JSON.parse(data.message),
          JSON.parse(data.response)
        );
      });
    },
    bindBotRegisterd() {
      this.pusher.channel.bind("bot-registered", (data: Bot) => {
        if (this.store.bots.length === 0) {
          const voices = this.synth.getVoices();
          const englishVoices = voices.filter((voice) => {
            return (
              voice.lang === "en-GB" ||
              voice.lang === "en-US" ||
              voice.lang === "en-CA" ||
              voice.lang === "en-AU"
            );
          });
          this.store.setVoices(englishVoices);
        }
        const bot = this.store.addBot(data);
        this.speak("Hello!", bot.voice);
      });
    },
  },
};
</script>

<template>
  <div
    class="flex flex-col w-full h-screen bg-gradient-to-bl from-cyan-700 to-teal-400 text-cyan-900"
  >
    <div class="w-full pt-1 px-2">
      <Navigation />
    </div>
    <div class="w-full flex-1 px-2 flex overflow-hidden">
      <div class="w-1/2 h-full py-2 pr-2">
        <div
          class="flex flex-col w-full h-full bg-white/40 rounded-lg p-2 shadow-lg overflow-hidden"
        >
          <h2 class="border-b pb-1 border-teal-600 font-light text-cyan-900">
            Chat
          </h2>
          <Chat />
        </div>
      </div>
      <div class="w-1/2 h-full">
        <div class="w-full h-1/3 py-2 pl-2">
          <div
            class="flex flex-col w-full h-full bg-white/40 rounded-lg p-2 shadow-lg overflow-hidden"
          >
            <h2 class="border-b pb-1 border-teal-600 font-light text-cyan-900">
              Connected Bots
            </h2>
            <Connections />
          </div>
        </div>
        <div class="w-full h-2/3 py-2 pl-2">
          <div
            class="flex flex-col w-full h-full bg-white/40 rounded-lg p-2 shadow-lg overflow-hidden"
          >
            <h2 class="border-b pb-1 border-teal-600 font-light text-cyan-900">
              Message Ranking
            </h2>
            <Rankings />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
