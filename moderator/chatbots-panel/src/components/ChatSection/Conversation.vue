<script lang="ts">
import Message from "./Message.vue";
import { useStore } from "@/stores/store";

export default {
  components: {
    Message,
  },
  setup() {
    const store = useStore();
    return { store };
  },
};
</script>

<template>
  <div id="conversation" class="overflow-scroll w-full h-full pr-4 pb-1">
    <div v-for="message in store.conversation">
      <Message :author="message.bot_name" :text="message.message" />
    </div>
  </div>
</template>
