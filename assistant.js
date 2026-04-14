const fs = require("fs");
const path = require("path");
const os = require("os");
const { exec, execSync, spawn } = require("child_process");

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function fetchWithTimeout(url, options = {}, timeoutMs = 5000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } catch (e) {
    if (e && (e.name === "AbortError" || /aborted/i.test(String(e.message || e)))) {
      throw new Error(`Request timed out after ${timeoutMs}ms`);
    }
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

function findFfmpegBinary() {
  const envPath = process.env.FFMPEG_PATH;
  if (envPath && fs.existsSync(envPath)) {
    return envPath;
  }

  const localCandidates = [
    path.resolve("ffmpeg.exe"),
    path.resolve("tools", "ffmpeg", "bin", "ffmpeg.exe"),
    path.resolve("models", "ffmpeg", "bin", "ffmpeg.exe")
  ];

  const localMatch = localCandidates.find(p => fs.existsSync(p));
  if (localMatch) {
    return localMatch;
  }

  // Common WinGet install location on Windows.
  const localAppData = process.env.LOCALAPPDATA;
  if (localAppData) {
    const wingetPackages = path.join(localAppData, "Microsoft", "WinGet", "Packages");
    if (fs.existsSync(wingetPackages)) {
      try {
        const packageDirs = fs.readdirSync(wingetPackages)
          .filter(name => name.toLowerCase().startsWith("gyan.ffmpeg_"));

        for (const dir of packageDirs) {
          const packageRoot = path.join(wingetPackages, dir);
          const nestedDirs = fs.readdirSync(packageRoot, { withFileTypes: true })
            .filter(entry => entry.isDirectory())
            .map(entry => entry.name);

          for (const nested of nestedDirs) {
            const candidate = path.join(packageRoot, nested, "bin", "ffmpeg.exe");
            if (fs.existsSync(candidate)) {
              return candidate;
            }
          }
        }
      } catch (_) {
        // Ignore directory traversal failures and keep trying other strategies.
      }
    }
  }

  try {
    const whereOutput = execSync("where ffmpeg", { stdio: ["ignore", "pipe", "ignore"] })
      .toString()
      .split(/\r?\n/)
      .map(s => s.trim())
      .find(Boolean);

    if (whereOutput && fs.existsSync(whereOutput)) {
      return whereOutput;
    }
  } catch (_) {
    // ffmpeg is not in PATH.
  }

  return null;
}

function listWindowsAudioDevices(ffmpeg) {
  return new Promise((resolve) => {
    exec(`"${ffmpeg}" -hide_banner -list_devices true -f dshow -i dummy`, (err, stdout, stderr) => {
      const output = `${stdout || ""}\n${stderr || ""}`;
      const devices = [];
      const lines = output.split(/\r?\n/);

      for (const line of lines) {
        if (!/\(audio\)/i.test(line)) {
          continue;
        }

        const match = line.match(/"([^"]+)"/);
        if (match && match[1]) {
          devices.push(match[1]);
        }
      }

      resolve([...new Set(devices)]);
    });
  });
}

function findWhisperCliBinary() {
  const envPath = process.env.WHISPER_CLI_PATH;
  if (envPath && fs.existsSync(envPath)) {
    return envPath;
  }

  const candidates = [
    path.resolve("models", "whisper-bin-x64", "Release", "whisper-cli.exe"),
    path.resolve("whisper-cli.exe")
  ];

  return candidates.find(p => fs.existsSync(p)) || null;
}

function findWhisperModelPath(whisperDir) {
  const envModelPath = process.env.WHISPER_MODEL_PATH;
  if (envModelPath && fs.existsSync(envModelPath)) {
    return envModelPath;
  }

  const preferredModels = [
    // base.en is English-only but significantly more accurate than tiny.
    "ggml-base.bin",
  ];

  for (const modelName of preferredModels) {
    const modelPath = path.join(whisperDir, modelName);
    if (fs.existsSync(modelPath)) {
      return modelPath;
    }
  }

  return null;
}

function findLlamaServerBinary() {
  const envPath = process.env.LLAMA_SERVER_PATH;
  if (envPath && fs.existsSync(envPath)) {
    return envPath;
  }

  const candidates = [
    path.resolve("llama.cpp", "build", "bin", "Release", "llama-server.exe"),
    path.resolve("llama-server.exe")
  ];

  return candidates.find(p => fs.existsSync(p)) || null;
}

function findLlamaModelPath() {
  const envModel = process.env.LLAMA_MODEL_PATH;
  if (envModel && fs.existsSync(envModel)) {
    return envModel;
  }

  const candidates = [
    path.resolve("models", "phi-3-mini-4k-instruct-q4_k_m.gguf")
  ];

  return candidates.find(p => fs.existsSync(p)) || null;
}

let llamaServerProcess = null;

function shouldKeepServerAlive() {
  // Default: keep llama-server alive so you don't pay cold-start cost each run.
  // Set KEEP_SERVER_ALIVE=0 to revert to previous behavior.
  const v = String(process.env.KEEP_SERVER_ALIVE ?? "1").trim().toLowerCase();
  return !(v === "0" || v === "false" || v === "no");
}

function getLlamaServerBaseUrl() {
  return (process.env.LLAMA_SERVER_URL || "http://127.0.0.1:8080").replace(/\/$/, "");
}

async function isLlamaServerUp() {
  const base = getLlamaServerBaseUrl();
  try {
    // llama-server serves a web UI on /, so any 2xx/3xx is fine.
    const res = await fetchWithTimeout(`${base}/`, { method: "GET" }, 1500);
    return !!res;
  } catch (_) {
    return false;
  }
}

async function ensureLlamaServerRunning() {
  if (await isLlamaServerUp()) {
    return;
  }

  const llamaServer = findLlamaServerBinary();
  if (!llamaServer) {
    throw new Error(
      "llama-server.exe not found. Build it in llama.cpp or set LLAMA_SERVER_PATH to its full path."
    );
  }

  const modelPath = findLlamaModelPath();
  if (!modelPath) {
    throw new Error(
      "No GGUF model found for llama-server. Put a model at models/phi-3-mini-4k-instruct-q4_k_m.gguf or set LLAMA_MODEL_PATH."
    );
  }

  const url = new URL(getLlamaServerBaseUrl());
  const host = url.hostname || "127.0.0.1";
  const port = Number(url.port || 8080);
  if (!Number.isFinite(port)) {
    throw new Error(`Invalid LLAMA_SERVER_URL port: ${url.port}`);
  }

  if (llamaServerProcess && !llamaServerProcess.killed) {
    // We started one before, but it isn't reachable yet.
  } else {
    const args = [
      "-m", modelPath,
      "--host", host,
      "--port", String(port)
    ];

    // Optional performance knobs (only applied when explicitly set).
    if (process.env.LLAMA_THREADS) {
      args.push("--threads", String(process.env.LLAMA_THREADS));
    }
    if (process.env.LLAMA_THREADS_HTTP) {
      args.push("--threads-http", String(process.env.LLAMA_THREADS_HTTP));
    }

    console.log(`Starting llama-server on ${host}:${port} ...`);

    // Detached keeps the server alive even if Node exits.
    // Note: stdio must be "ignore" for reliable detaching on Windows.
    llamaServerProcess = spawn(llamaServer, args, {
      detached: true,
      stdio: "ignore",
      windowsHide: true
    });

    llamaServerProcess.unref();
  }

  const start = Date.now();
  const timeoutMs = 20000;
  while (Date.now() - start < timeoutMs) {
    if (await isLlamaServerUp()) {
      return;
    }
    await sleep(300);
  }

  throw new Error("llama-server did not become ready in time.");
}

function findPiperBinary() {
  const envPath = process.env.PIPER_PATH;
  if (envPath && fs.existsSync(envPath)) {
    return envPath;
  }

  const candidates = [
    path.resolve("piper", "piper.exe"),
    path.resolve("piper.exe")
  ];

  return candidates.find(p => fs.existsSync(p)) || null;
}

function findPiperVoiceModel() {
  const envModel = process.env.PIPER_VOICE_PATH;
  if (envModel && fs.existsSync(envModel)) {
    return envModel;
  }

  const candidates = [
    path.resolve("piper", "voices", "en_US-lessac-medium.onnx")
  ];

  return candidates.find(p => fs.existsSync(p)) || null;
}

function findFfplayBinary() {
  const ffmpeg = findFfmpegBinary();
  if (ffmpeg) {
    const ffplay = path.join(path.dirname(ffmpeg), "ffplay.exe");
    if (fs.existsSync(ffplay)) {
      return ffplay;
    }
  }

  const candidates = [
    path.resolve("ffplay.exe"),
    path.resolve("tools", "ffmpeg", "bin", "ffplay.exe"),
    path.resolve("models", "ffmpeg", "bin", "ffplay.exe")
  ];
  return candidates.find(p => fs.existsSync(p)) || null;
}

function playWav(wavPath) {
  return new Promise((resolve, reject) => {
    const ffplay = findFfplayBinary();
    if (ffplay) {
      const child = spawn(ffplay, ["-nodisp", "-autoexit", "-loglevel", "error", wavPath], {
        stdio: "ignore",
        windowsHide: true
      });
      child.on("error", (e) => reject(e));
      child.on("exit", () => resolve());
      return;
    }

    // Fallback: Windows SoundPlayer via PowerShell.
    const ps = spawn(
      "powershell",
      [
        "-NoProfile",
        "-Command",
        `Add-Type -AssemblyName System.Windows.Forms;` +
          `$p = New-Object System.Media.SoundPlayer '${wavPath.replace(/'/g, "''")}';` +
          `$p.PlaySync();`
      ],
      { stdio: "ignore", windowsHide: true }
    );
    ps.on("error", (e) => reject(e));
    ps.on("exit", () => resolve());
  });
}

async function speakWithPiper(text) {
  const piper = findPiperBinary();
  const voice = findPiperVoiceModel();

  if (!piper || !voice) {
    console.log("(TTS skipped: Piper not found. Set PIPER_PATH and PIPER_VOICE_PATH to enable voice replies.)");
    return;
  }

  const outWav = path.resolve(`reply-${Date.now()}.wav`);
  await new Promise((resolve, reject) => {
    const child = spawn(piper, ["--model", voice, "--output_file", outWav], {
      stdio: ["pipe", "ignore", "pipe"],
      windowsHide: true
    });

    child.on("error", (e) => reject(e));
    child.stderr.on("data", (chunk) => {
      const msg = String(chunk || "").trim();
      if (msg) {
        console.error(msg);
      }
    });

    child.stdin.write(String(text || ""));
    child.stdin.end();

    child.on("exit", (code) => {
      if (code === 0 && fs.existsSync(outWav)) {
        resolve();
      } else {
        reject(new Error(`Piper failed (exit code ${code}).`));
      }
    });
  });

  try {
    await playWav(outWav);
  } finally {
    try { fs.unlinkSync(outWav); } catch (_) {}
  }
}

function speechToText(audio) {
  return new Promise((resolve, reject) => {
    const whisperCli = findWhisperCliBinary();
    if (!whisperCli) {
      reject(new Error("Whisper CLI not found. Set WHISPER_CLI_PATH or place whisper-cli.exe in models/whisper-bin-x64/Release."));
      return;
    }

    const whisperCwd = path.dirname(whisperCli);
    const modelPath = findWhisperModelPath(whisperCwd);
    if (!modelPath) {
      reject(new Error("No Whisper model found. Set WHISPER_MODEL_PATH or add a ggml-*.bin model to models/whisper-bin-x64/Release."));
      return;
    }

    const audioAbs = path.resolve(audio);

    const threads = Math.max(1, Math.min(8, Number(process.env.WHISPER_THREADS || os.cpus().length || 4)));
    if (!process.env.QUIET_WHISPER) {
      console.log(`Whisper: model=${path.basename(modelPath)} threads=${threads}`);
    }

    // Use -of to write a guaranteed .txt output file — more reliable than
    // parsing stdout which varies between whisper-cli builds.
    const outBase = audioAbs.replace(/\.[^.]+$/, "");
    const outTxt = outBase + ".txt";

    // Clean up any stale output file from a previous run.
    try { if (fs.existsSync(outTxt)) fs.unlinkSync(outTxt); } catch (_) {}

    const cmd = [
      `"${whisperCli}"`,
      "-m", `"${modelPath}"`,
      "-f", `"${audioAbs}"`,
      "-t", String(threads),
      "-l", "en",
      "-bs", "1",
      "-bo", "1",
      "-nt",
      "-of", `"${outBase}"`
    ].join(" ");

    exec(
      cmd,
      { cwd: whisperCwd },
      (err, stdout, stderr) => {
        if (err) {
          console.error("Whisper failed:", stderr || err.message);
        }

        // Prefer the .txt file whisper-cli writes via -of.
        let text = "";
        if (fs.existsSync(outTxt)) {
          text = fs.readFileSync(outTxt, "utf8");
        }

        // Fall back to stdout if the file wasn't created.
        if (!text.trim()) {
          text = stdout || "";
        }

        text = text
          .split("\n")
          .filter(l => l.trim())
          .join(" ")
          // Strip whisper metadata: [00:00 --> 00:05], [BLANK_AUDIO], etc.
          .replace(/\[[^\]]+\]/g, "")
          // Strip whisper hallucinations with parentheses: (speaking in foreign language)
          .replace(/\([^)]+\)/g, "")
          .replace(/\s+/g, " ")
          .trim();

        resolve(text);
      }
    );
  });
}

async function askAI(prompt) {
  await ensureLlamaServerRunning();

  const base = getLlamaServerBaseUrl();
  const requestTimeoutMs = Number(process.env.LLAMA_REQUEST_TIMEOUT_MS || 120000);
  const system = "You are a friendly desk robot assistant. Reply in one short sentence.";

  // Prefer OpenAI-compatible endpoint.
  try {
    const res = await fetchWithTimeout(
      `${base}/v1/chat/completions`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: [
            { role: "system", content: system },
            { role: "user", content: prompt }
          ],
          max_tokens: Number(process.env.LLAMA_MAX_TOKENS || 32),
          temperature: 0.2
        })
      },
      requestTimeoutMs
    );

    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(`HTTP ${res.status}: ${body}`);
    }

    const data = await res.json();
    const content = data?.choices?.[0]?.message?.content;
    if (typeof content === "string") {
      return content.replace(/\s+/g, " ").trim();
    }
  } catch (e) {
    // Fall back to legacy /completion if enabled in this build.
    const errMsg = e?.message || String(e);
    console.warn("Chat API failed, trying legacy /completion:", errMsg);
  }

  const legacyRes = await fetchWithTimeout(
    `${base}/completion`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: `<|system|>\n${system}\n\n<|user|>\n${prompt}\n\n<|assistant|>`,
        n_predict: Number(process.env.LLAMA_N_PREDICT || 32),
        temperature: 0.2,
        stop: ["<|assistant|>", "<|user|>", "<|system|>"]
      })
    },
    requestTimeoutMs
  );

  if (!legacyRes.ok) {
    const body = await legacyRes.text().catch(() => "");
    throw new Error(`LLM request failed (HTTP ${legacyRes.status}): ${body}`);
  }

  const legacyData = await legacyRes.json();
  return (legacyData.content || "")
    .replace(/<\|[^|]+\|>/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function recordAudio() {
  return new Promise((resolve, reject) => {
    // Default shorter max length for responsiveness.
    const maxRecordSeconds = Number(process.env.MAX_RECORD_SECONDS || 8);
    // -40dB is the right sensitivity: detects quiet speech while ignoring
    // very low-level electrical hiss. -25dB is too strict (only very loud
    // sounds qualify), -35dB was the original and also works.
    const silenceNoiseDb = String(process.env.SILENCE_NOISE_DB || "-40dB");
    const silenceDetectSeconds = Number(process.env.SILENCE_DETECT_SECONDS || 0.20);
    const stopAfterSilenceMs = Number(process.env.STOP_AFTER_SILENCE_MS || 600);

    console.log("🎤 Listening... (pause to send)");

    const ffmpeg = findFfmpegBinary();
    if (!ffmpeg) {
      const msg = [
        "ffmpeg not found.",
        "Install it with: winget install Gyan.FFmpeg",
        "or set FFMPEG_PATH to your ffmpeg.exe location."
      ].join(" ");
      reject(new Error(msg));
      return;
    }

    const output = path.resolve("input.wav");

    const runRecord = (micName) => {
      // Use ffmpeg's silencedetect to stop early after you pause speaking.
      // This keeps turn-taking responsive without needing push-to-talk.
      const args = [
        "-hide_banner",
        "-loglevel", "info",
        "-f", "dshow",
        "-i", `audio=${micName}`,
        "-t", String(maxRecordSeconds),
        "-ac", "1",
        "-ar", "16000",
        "-acodec", "pcm_s16le",
        "-af", `volume=4,silencedetect=n=${silenceNoiseDb}:d=${silenceDetectSeconds}`,
        "-y", output
      ];

      const child = spawn(ffmpeg, args, { windowsHide: true });

      let stderrBuffer = "";
      let stopTimer = null;
      let minTimeElapsed = false;

      // Wait at least this long before allowing silence to stop the recording.
      // This prevents the recording from cutting off before the user starts speaking.
      const minRecordMs = Number(process.env.MIN_RECORD_MS || 1500);
      const minTimer = setTimeout(() => { minTimeElapsed = true; }, minRecordMs);

      const clearStopTimer = () => {
        if (stopTimer) {
          clearTimeout(stopTimer);
          stopTimer = null;
        }
      };

      child.on("error", (e) => {
        clearTimeout(minTimer);
        clearStopTimer();
        reject(new Error(`Recording failed: ${e.message || e}`));
      });

      if (child.stderr) {
        child.stderr.on("data", (chunk) => {
          const text = String(chunk || "");
          stderrBuffer += text;

          // Cancel any pending early-stop when speech resumes.
          if (/silence_end:/i.test(text)) {
            clearStopTimer();
          }

          // After the minimum recording time, any detected pause ends the
          // recording. We do NOT require silence_end to have fired first:
          // on quiet mics silence_end may never fire (signal never crosses
          // the threshold upward), so waiting for it would hang until the
          // 8-second cap every time.
          if (/silence_start:/i.test(text) && minTimeElapsed && !stopTimer) {
            stopTimer = setTimeout(() => {
              try { child.kill(); } catch (_) {}
            }, stopAfterSilenceMs);
          }
        });
      }

      child.on("exit", (code) => {
        clearTimeout(minTimer);
        clearStopTimer();

        if (code !== 0 && code !== null) {
          const details = stderrBuffer.trim();
          if (/Could not find audio device|No audio devices|I\/O error/i.test(details)) {
            reject(new Error(
              "Recording failed: microphone device was not found. " +
              "Set MIC_DEVICE to one of your Windows audio device names and retry."
            ));
            return;
          }

          // ffmpeg may return non-zero when we kill it early; accept the file if it exists.
          if (!fs.existsSync(output)) {
            reject(new Error(`Recording failed: ${details || `ffmpeg exit code ${code}`}`));
            return;
          }
        }

        console.log("Recording finished");
        resolve(output);
      });
    };

    const requestedMic = process.env.MIC_DEVICE;
    if (requestedMic) {
      runRecord(requestedMic);
      return;
    }

    listWindowsAudioDevices(ffmpeg)
      .then((devices) => {
        if (!devices.length) {
          reject(new Error("Recording failed: no Windows audio input devices were detected by ffmpeg."));
          return;
        }

        const preferred = devices.find(d => /microphone|mic/i.test(d)) || devices[0];
        console.log(`Using microphone: ${preferred}`);
        runRecord(preferred);
      })
      .catch((e) => {
        reject(new Error(`Recording failed: ${e.message || e}`));
      });
  });
}

async function greet() {
  const greeting = "Hello! I am your robot assistant. How can I help you today?";
  console.log(`\nAssistant: ${greeting}`);
  await speakWithPiper(greeting);
}

async function run() {
  process.on("SIGINT", () => {
    console.log("\nStopping...");
    try {
      if (!shouldKeepServerAlive()) {
        if (llamaServerProcess && !llamaServerProcess.killed) {
          llamaServerProcess.kill();
        }
      } else {
        console.log("(Keeping llama-server running)");
      }
    } catch (_) {}
    process.exit(0);
  });

  await greet();

  while (true) {
    try {
      const audio = await recordAudio();
      const text = await speechToText(audio);

      console.log("You said:", text || "(nothing detected)");

      if (!text.trim()) {
        console.log("AI: I couldn't catch that. Please try again.");
        await sleep(250);
        continue;
      }

      const reply = await askAI(text);
      const finalReply = reply || "I am not sure how to respond yet.";
      console.log("AI:", finalReply);
      await speakWithPiper(finalReply);
      await sleep(250);
    } catch (err) {
      console.error("Assistant failed:", err.message || err);
      await sleep(1000);
    }
  }
}

run();