<script lang="ts">
export default {
  components: {},
  setup() {},
  data() {
    return {
      input: this.$refs.input,
      theoTyping: false,
      conversation: [
        {
          theo: true,
          text: 'Hey there! I am Theo, a transformer-based chatbot.',
        },
        {
          theo: true,
          text: 'Feel free to ask me anything!',
        },
      ],
    };
  },
  methods: {
    chat(evt: Event) {
      const conversationContainer = <HTMLElement> document.getElementById("conversation-container");
      const form = <HTMLFormElement> evt.target;
      const data = new FormData(form);
      form.reset();
      setTimeout(() => {
          conversationContainer.scrollTo({
              top: conversationContainer.scrollHeight,
              left: 0,
              behavior: "smooth",
            });
        }, 200);
      setTimeout(() => {
        this.theoTyping = true;
      }, 100)
      this.conversation.push({
        theo: false,
        text: <string> data.get("message"),
      })
      fetch("http://127.0.0.1:4000/chat", {
        method: "POST",
        headers: {
          'Accept': 'application/json',
        },
        body: data
      })
      .then(res => res.json())
      .then((data) => { 
        this.theoTyping = false;
        this.conversation.push({
          theo: true,
          text: data,
        });
        
        setTimeout(() => {
          conversationContainer.scrollTo({
              top: conversationContainer.scrollHeight,
              left: 0,
              behavior: "smooth",
            });
        }, 200);
      });
    },
  }
};
</script>

<template>
  <div class="flex flex-col flex-1 justify-between items-between overflow-hidden">
    <div id="conversation-container" class="p-3 pb-10 overflow-scroll">
      <div v-for="message, index in conversation" class="text-left">
        <div v-if="message.theo" class="w-full flex flex-col">
          <div v-if="index === 0 || !conversation[index-1].theo" class="text-[12px] font-bold">Theo</div>
          <div class="min-w-[20%] max-w-[80%] rounded-md bg-[#d2fbd0] p-2 text-[15px] font-bold" :class="{'mb-2': index !== conversation.length-1}">
            <p class="text-[#0d5f07]">{{message.text}}</p>
          </div>
        </div>
        <div v-else class="w-full flex items-end flex-col">
          <div v-if="index === 0 || conversation[index-1].theo" class="text-[12px] w-full text-right font-bold">You</div>
          <div class="min-w-[20%] max-w-[80%] rounded-md bg-[#acc8e5] p-2 text-[15px] font-bold" :class="{'mb-2': index !== conversation.length-1}">
            <p class="text-[#112a46]">{{message.text}}</p>
          </div>
        </div>
      </div>
    </div>
    <div>
      <div class="w-full flex items-start">
        <div v-if="theoTyping" class="flex items-center">
          <p>Theo is typing </p>
          <div class="ticontainer ml-1">
            <div class="tiblock">
              <div class="tidot"></div>
              <div class="tidot"></div>
              <div class="tidot"></div>
            </div>
          </div>
        </div>
      </div>
      <div class="w-full my-3">
        <form @submit.prevent="chat">
          <input ref="submit" name="message" class="w-full rounded-md text-[15px] focus:outline-none focus:border:none p-2" type="text"/>
        </form>
      </div>
    </div>
  </div>
</template>
