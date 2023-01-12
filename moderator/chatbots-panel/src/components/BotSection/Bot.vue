<script lang="ts">
export default {
  setup() {},
  props: {
    name: "",
    id: "",
    voice: "",
  },
};
</script>

<template>
  <div class="w-full flex flex-col my-2 py-1 px-2 rounded-lg bg-slate-700">
    <b class="text-[12px] text-white">{{ name }}</b>
    <small class="text-[12] text-white">{{ id }}</small>
    <small class="text-[12] text-white">{{ voice }}</small>
  </div>
</template>
