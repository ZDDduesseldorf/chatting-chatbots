const synth = window.speechSynthesis;

function speak(text, voice = false) {
  const utterance = new SpeechSynthesisUtterance(text);
  if (voice) {
    utterance.voice = voice;
  }
  utterance.rate = 0.7;
  synth.speak(utterance);
  return utterance;
}
