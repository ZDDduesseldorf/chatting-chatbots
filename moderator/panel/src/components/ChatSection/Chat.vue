<script lang="ts">
import Conversation from "./Conversation.vue";
import { usePusher } from "@/stores/pusher";
import { useStore } from "@/stores/store";

export default {
  components: {
    Conversation,
  },
  setup() {
    const pusher = usePusher();
    const store = useStore();
    return { pusher, store };
  },
  data() {
    return {
      input: "",
    };
  },
  methods: {
    sendMessage() {
      this.pusher.channel.trigger("client-provoke-message", this.input);
      this.input = "";
    },
  },
};
</script>

<template>
  <div class="flex flex-col h-full justify-between overflow-hidden">
    <div class="overflow-hidden">
      <Conversation />
    </div>
    <div class="pt-2 border-t border-teal-600">
      <textarea
        v-model="input"
        class="w-full h-[80px] p-1 border rounded-lg resize-none outline-none text-cyan-900 placeholder-cyan-900/50 border-teal-600 bg-transparent"
        type="text"
        placeholder="Message"
      />
      <button
        @click="sendMessage"
        class="absolute right-[5px] bottom-[10px] w-[80px] text-[12px] font-bold bg-slate-700 p-1 rounded-md text-white cursor-pointer"
      >
        Send
      </button>
    </div>
  </div>
</template>
