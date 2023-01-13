<script lang="ts">
import { Icon } from "@iconify/vue";
import { useStore } from "@/stores/store";

export default {
  components: {
    Icon,
  },
  setup() {
    const store = useStore();
    return { store };
  },
};
</script>

<template>
  <div class="h-full flex justify-around">
    <div
      @click="store.downloadRankings()"
      class="flex items-center justify-center rounded-md bg-slate-700 mr-4 w-[30px] h-[30px] cursor-pointer hover:scale-110"
    >
      <Icon icon="mdi:download" class="text-[25px] text-white" />
    </div>
    <div
      class="flex items-center justify-center rounded-md bg-slate-700 mr-4 w-[30px] h-[30px] cursor-pointer hover:scale-110"
    >
      <Icon icon="mdi:pause" class="text-[25px] text-white" />
    </div>
    <div
      class="flex items-center justify-center w-[30px] h-[30px] rounded-md bg-red-600 cursor-pointer hover:scale-110"
    >
      <Icon icon="mdi:stop" class="text-[25px] text-white" />
    </div>
  </div>
</template>
