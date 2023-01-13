<script setup lang="ts">
import Pusher from "pusher-js";
import * as PusherTypes from "pusher-js";
import Panel from "./components/Panel.vue";
import { useStore } from "@/stores/store";
import { usePusher } from "@/stores/pusher";

const store = useStore();
const pusher = usePusher();

pusher.setClient(
  new Pusher("66736225056eacd969c1", {
    cluster: "eu",
    channelAuthorization: {
      endpoint: "http://127.0.0.1:3000/pusher/auth",
    },
  })
);
pusher.setChannel(pusher.client.subscribe("private-panel"));
</script>

<template>
  <main>
    <Panel />
  </main>
</template>
