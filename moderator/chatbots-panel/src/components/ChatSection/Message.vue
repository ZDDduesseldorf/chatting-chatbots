<script lang="ts">
import Message from "./Message.vue";
import { useStore } from "@/stores/store";

export default {
  setup() {},
  props: {
    author: "",
    text: "",
  },
};
</script>

<template>
  <div class="my-1">
    <p class="text-cyan-900">
      <b>{{ author }}: </b>{{ text }}
    </p>
  </div>
</template>
