const WebSocket = require("ws");
const mic = require("mic");

const ws = new WebSocket("ws://localhost:3001");

ws.on("open", () => {
  console.log("🎤 Connected to AI server");

  const micInstance = mic({
    rate: "16000",
    channels: "1",
    debug: false,
    exitOnSilence: 0
  });

  const stream = micInstance.getAudioStream();

  stream.on("data", (chunk) => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(chunk); // 🔥 live audio streaming
    }
  });

  micInstance.start();
});

ws.on("message", async (data) => {
  const msg = JSON.parse(data.toString());

  if (msg.type === "text") {
    console.log("🤖 AI:", msg.data);
  }
});