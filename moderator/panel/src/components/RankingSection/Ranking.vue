<script lang="ts">
import { Icon } from "@iconify/vue";

export default {
  components: {
    Icon,
  },
  setup() {},
  props: {
    message: null,
    rank: null,
  },
};
</script>

<template>
  <div class="w-full flex flex-col my-2 py-1 px-2 rounded-lg bg-slate-700">
    <div class="text-[12px] text-white flex items-center justify-between">
      <div class="flex items-center">
        <div class="w-[50px] text-center">
          <b>{{ message.ranking_number.toFixed(2) }}</b>
        </div>
        <div class="pl-1">
          <div>
            <p class="text-[12px] text-white">
              <b>{{ message.bot_name }}: {{ message.message }}</b>
            </p>
          </div>
          <div>
            <small class="text-[12px] text-white">
              Similarity: {{ message.similarity_score.toFixed(2) }} - Share:
              {{ message.share_score.toFixed(2) }} - Topic:
              {{ message.topic_score.toFixed(2) }}</small
            >
          </div>
        </div>
      </div>
      <Icon icon="mdi:crown" class="text-[25px]" v-if="rank == 1" />
    </div>
  </div>
</template>
