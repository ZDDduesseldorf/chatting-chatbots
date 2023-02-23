<script setup lang="ts">
import Chat from './components/Chat.vue'
</script>

<template>
  <div class="h-full p-5 flex flex-col">
    <div>
      <h1 class="text-[5rem] m-5 font-bold flex mt-0">
        <div class="text-[#112A46] bg-[#acc8e5] rounded-md px-4 mr-4 ml-1">Chat</div> with <div class="text-[#0d5f07] bg-[#d2fbd0] rounded-md px-4 ml-4 mr-1">Theo</div>.
      </h1>
    </div>
    <Chat/>
  </div>
</template>