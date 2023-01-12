import { defineStore } from 'pinia';

export type PusherState = {
  channel: object;
  client: object;
  authToken: string;
};

export const usePusher = defineStore('pusher', {
  state: () => ({
    channel: {},
    client: {},
  } as PusherState),
  getters: {},
  actions: {
    setChannel(channel: object) {
      this.channel = channel
    },
    setClient(client: object) {
      this.client = client
    },
  },
})