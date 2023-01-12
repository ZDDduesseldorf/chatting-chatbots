<script lang="ts">
import Bot from "./Bot.vue";
import { useStore } from "@/stores/store";

export default {
  components: {
    Bot,
  },
  setup() {
    const store = useStore();
    return { store };
  },
};
</script>
<template>
  <div class="overflow-scroll w-full h-full pr-4">
    <div v-for="bot in store.bots">
      <Bot :name="bot.name" :id="bot.id" :voice="bot.voice.name" />
    </div>
  </div>
</template>
