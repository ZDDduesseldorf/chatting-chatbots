<script lang="ts">
import Ranking from "./Ranking.vue";
import { useStore } from "@/stores/store";

export default {
  components: {
    Ranking,
  },
  setup() {
    const store = useStore();
    return { store };
  },
};
</script>
<template>
  <div id="rankings" class="overflow-scroll w-full h-full pr-4">
    <div v-for="ranking in store.rankings" class="my-1">
      <div class="text-[12px] text-cyan-900">
        <b>{{ ranking.message.bot_name }}: </b>{{ ranking.message.message }}
      </div>
      <div v-for="(response, index) in ranking.ranked_responses">
        <Ranking :message="response" :rank="index + 1" />
      </div>
    </div>
  </div>
</template>
