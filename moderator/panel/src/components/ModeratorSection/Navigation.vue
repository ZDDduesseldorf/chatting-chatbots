<script lang="ts">
import Actions from "./Actions.vue";
export default {
  components: {
    Actions,
  },
  setup() {},
};
</script>

<template>
  <div class="w-full h-full flex justify-between px-2 items-center">
    <h1 class="text-[30px] font-extrabold text-white">Chatting Chatbots</h1>
    <div>
      <Actions />
    </div>
  </div>
</template>
