const state = {
  videos: [],
  selectedVideoId: localStorage.getItem("selectedVideoId") || "",
  activeTab: localStorage.getItem("activeTab") || "chat",
  chatByVideo: loadJson("chatByVideo", {}),
  activityTimerId: null,
  autoRefreshTimerId: null,
  refreshInFlight: false,
};

const elements = {
  activityBanner: document.getElementById("activity-banner"),
  tabs: Array.from(document.querySelectorAll(".nav-button")),
  tabChat: document.getElementById("tab-chat"),
  tabLibrary: document.getElementById("tab-library"),
  videoSelect: document.getElementById("video-select"),
  stats: document.getElementById("stats"),
  statusText: document.getElementById("status-text"),
  videoMeta: document.getElementById("video-meta"),
  videoPlayer: document.getElementById("video-player"),
  generateSummary: document.getElementById("generate-summary"),
  summaryActivity: document.getElementById("summary-activity"),
  summaryContent: document.getElementById("summary-content"),
  chatCount: document.getElementById("chat-count"),
  chatActivity: document.getElementById("chat-activity"),
  chatMessages: document.getElementById("chat-messages"),
  chatForm: document.getElementById("chat-form"),
  questionInput: document.getElementById("question-input"),
  clearChat: document.getElementById("clear-chat"),
  uploadForm: document.getElementById("upload-form"),
  uploadFile: document.getElementById("upload-file"),
  uploadProfile: document.getElementById("upload-profile"),
  uploadLanguage: document.getElementById("upload-language"),
  libraryCount: document.getElementById("library-count"),
  libraryList: document.getElementById("library-list"),
  chatTemplate: document.getElementById("chat-message-template"),
  libraryTemplate: document.getElementById("library-card-template"),
};

function loadJson(key, fallback) {
  try {
    const value = localStorage.getItem(key);
    return value ? JSON.parse(value) : fallback;
  } catch {
    return fallback;
  }
}

function saveJson(key, value) {
  localStorage.setItem(key, JSON.stringify(value));
}

function setStatus(text, isError = false) {
  elements.statusText.textContent = text;
  elements.statusText.style.color = isError ? "#7a1111" : "";
}

function showActivity(message, persist = true) {
  if (state.activityTimerId) {
    window.clearTimeout(state.activityTimerId);
    state.activityTimerId = null;
  }
  elements.activityBanner.textContent = message;
  elements.activityBanner.classList.remove("hidden");
  if (!persist) {
    state.activityTimerId = window.setTimeout(() => {
      elements.activityBanner.classList.add("hidden");
      state.activityTimerId = null;
    }, 2200);
  }
}

function hideActivity() {
  if (state.activityTimerId) {
    window.clearTimeout(state.activityTimerId);
    state.activityTimerId = null;
  }
  elements.activityBanner.classList.add("hidden");
}

function setInlineActivity(target, message = "") {
  if (!target) {
    return;
  }
  if (!message) {
    target.textContent = "";
    target.classList.add("hidden");
    return;
  }
  target.textContent = message;
  target.classList.remove("hidden");
}

function setBusy(target, busy) {
  if (!target) {
    return;
  }
  target.disabled = busy;
}

function formatDuration(seconds) {
  if (seconds === null || seconds === undefined) {
    return "—";
  }
  const total = Math.max(0, Math.floor(Number(seconds)));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours > 0) {
    return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  }
  return `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function getSelectedVideo() {
  return state.videos.find((video) => video.video_id === state.selectedVideoId) || null;
}

function getIndexedVideos() {
  return state.videos.filter((video) => video.status === "indexed");
}

function getChatItems(videoId) {
  if (!videoId) {
    return [];
  }
  state.chatByVideo[videoId] ||= [];
  return state.chatByVideo[videoId];
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function applyInlineMarkdown(text) {
  let value = escapeHtml(text);
  value = value.replace(/`([^`]+)`/g, "<code>$1</code>");
  value = value.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  value = value.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  value = value.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer noopener">$1</a>');
  return value;
}

function markdownToHtml(markdown) {
  const source = String(markdown || "").replace(/\r\n/g, "\n");
  const lines = source.split("\n");
  const html = [];
  let inList = false;
  let inCode = false;
  let codeLines = [];
  let paragraph = [];

  function flushParagraph() {
    if (!paragraph.length) {
      return;
    }
    html.push(`<p>${paragraph.map((line) => applyInlineMarkdown(line)).join("<br>")}</p>`);
    paragraph = [];
  }

  function closeList() {
    if (!inList) {
      return;
    }
    html.push("</ul>");
    inList = false;
  }

  lines.forEach((rawLine) => {
    const line = rawLine.replace(/\t/g, "    ");
    const trimmed = line.trim();

    if (trimmed.startsWith("```")) {
      closeList();
      flushParagraph();
      if (inCode) {
        html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
        codeLines = [];
        inCode = false;
      } else {
        inCode = true;
      }
      return;
    }

    if (inCode) {
      codeLines.push(rawLine);
      return;
    }

    if (!trimmed) {
      closeList();
      flushParagraph();
      return;
    }

    if (/^---+$/.test(trimmed) || /^\*\*\*+$/.test(trimmed)) {
      closeList();
      flushParagraph();
      html.push("<hr>");
      return;
    }

    const headingMatch = trimmed.match(/^(#{1,4})\s+(.*)$/);
    if (headingMatch) {
      closeList();
      flushParagraph();
      const level = headingMatch[1].length;
      html.push(`<h${level}>${applyInlineMarkdown(headingMatch[2])}</h${level}>`);
      return;
    }

    const quoteMatch = trimmed.match(/^>\s?(.*)$/);
    if (quoteMatch) {
      closeList();
      flushParagraph();
      html.push(`<blockquote>${applyInlineMarkdown(quoteMatch[1])}</blockquote>`);
      return;
    }

    const listMatch = trimmed.match(/^[-*]\s+(.*)$/);
    if (listMatch) {
      flushParagraph();
      if (!inList) {
        html.push("<ul>");
        inList = true;
      }
      html.push(`<li>${applyInlineMarkdown(listMatch[1])}</li>`);
      return;
    }

    closeList();
    paragraph.push(trimmed);
  });

  closeList();
  flushParagraph();
  if (inCode) {
    html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
  }

  return html.join("");
}

function renderMarkdown(target, markdown) {
  target.classList.add("markdown");
  target.innerHTML = markdownToHtml(markdown);
}

function renderPlain(target, text) {
  target.classList.remove("markdown");
  target.textContent = text;
}

function persistChat() {
  saveJson("chatByVideo", state.chatByVideo);
}

function selectTab(tab) {
  state.activeTab = tab;
  localStorage.setItem("activeTab", tab);
  elements.tabChat.classList.toggle("active", tab === "chat");
  elements.tabLibrary.classList.toggle("active", tab === "library");
  elements.tabs.forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tab);
  });
}

function updateStats() {
  const indexed = state.videos.filter((video) => video.status === "indexed").length;
  const failed = state.videos.filter((video) => video.status === "failed").length;
  const processing = state.videos.filter((video) => video.status === "processing").length;
  elements.stats.innerHTML = "";
  [
    ["Всего", state.videos.length],
    ["Проиндексировано", indexed],
    ["В процессе", processing],
    ["Не удалось проиндексировать", failed],
  ].forEach(([label, value]) => {
    const item = document.createElement("div");
    item.className = "stat-item";
    item.innerHTML = `<span>${label}</span><strong>${value}</strong>`;
    elements.stats.appendChild(item);
  });
}

function populateVideoSelect() {
  const indexedVideos = getIndexedVideos();
  const previous = state.selectedVideoId;
  elements.videoSelect.innerHTML = "";

  if (!indexedVideos.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "Нет проиндексированных видео";
    elements.videoSelect.appendChild(option);
    state.selectedVideoId = "";
    localStorage.removeItem("selectedVideoId");
    return;
  }

  if (!indexedVideos.some((video) => video.video_id === previous)) {
    state.selectedVideoId = indexedVideos[0].video_id;
  }

  indexedVideos.forEach((video) => {
    const option = document.createElement("option");
    option.value = video.video_id;
    option.textContent = `${video.title} · ${video.video_id}`;
    option.selected = video.video_id === state.selectedVideoId;
    elements.videoSelect.appendChild(option);
  });

  localStorage.setItem("selectedVideoId", state.selectedVideoId);
}

function renderChat() {
  const video = getSelectedVideo();
  const chatItems = getChatItems(state.selectedVideoId);
  elements.chatMessages.innerHTML = "";
  elements.chatCount.textContent = `${chatItems.length} сообщений`;

  if (!video) {
    elements.chatMessages.className = "chat-messages empty";
    elements.chatMessages.textContent = "Сначала выберите проиндексированное видео.";
    elements.videoMeta.textContent = "Нет выбранного видео";
    elements.videoPlayer.removeAttribute("src");
    elements.videoPlayer.load();
    setInlineActivity(elements.chatActivity);
    return;
  }

  elements.chatMessages.className = "chat-messages";
  elements.videoMeta.textContent = `${video.status} · ${formatDuration(video.duration_seconds)} · ${video.language || "язык не определён"}`;
  const nextVideoSrc = `/api/videos/${encodeURIComponent(video.video_id)}/file`;
  const currentVideoSrc = elements.videoPlayer.getAttribute("src") || "";
  if (currentVideoSrc !== nextVideoSrc) {
    elements.videoPlayer.src = nextVideoSrc;
  }

  if (!chatItems.length) {
    elements.chatMessages.className = "chat-messages empty";
    elements.chatMessages.textContent = "Задайте первый вопрос по видео.";
    return;
  }

  chatItems.forEach((item) => {
    const fragment = elements.chatTemplate.content.cloneNode(true);
    const root = fragment.querySelector(".message");
    const role = fragment.querySelector(".message-role");
    const content = fragment.querySelector(".message-content");
    const citations = fragment.querySelector(".message-citations");
    root.classList.add(item.role);
    role.textContent = item.role === "user" ? "Вы" : "Ассистент";
    if (item.role === "assistant") {
      renderMarkdown(content, item.content);
    } else {
      renderPlain(content, item.content);
    }
    if (item.role === "assistant" && Array.isArray(item.citations)) {
      item.citations.forEach((citation) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "secondary";
        button.textContent = `▶ ${citation}`;
        button.addEventListener("click", () => jumpToCitation(citation));
        citations.appendChild(button);
      });
    }
    elements.chatMessages.appendChild(fragment);
  });
  elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

async function renderSummary() {
  const video = getSelectedVideo();
  if (!video) {
    elements.summaryContent.className = "summary-content empty";
    renderPlain(elements.summaryContent, "Сначала выберите видео.");
    setInlineActivity(elements.summaryActivity);
    return;
  }

  try {
    const response = await fetch(`/api/videos/${encodeURIComponent(video.video_id)}/summary`);
    const data = await handleJson(response);
    if (!data.summary) {
      elements.summaryContent.className = "summary-content empty";
      renderPlain(elements.summaryContent, "Конспект ещё не сгенерирован.");
      return;
    }
    elements.summaryContent.className = "summary-content";
    renderMarkdown(elements.summaryContent, data.summary);
  } catch (error) {
    elements.summaryContent.className = "summary-content empty";
    renderPlain(elements.summaryContent, error.message);
  }
}

function renderLibrary() {
  elements.libraryCount.textContent = String(state.videos.length);
  elements.libraryList.innerHTML = "";
  if (!state.videos.length) {
    elements.libraryList.className = "library-list empty";
    elements.libraryList.textContent = "Библиотека пуста.";
    return;
  }

  elements.libraryList.className = "library-list";
  state.videos.forEach((video) => {
    const fragment = elements.libraryTemplate.content.cloneNode(true);
    const card = fragment.querySelector(".library-card");
    const title = fragment.querySelector(".library-title");
    const meta = fragment.querySelector(".library-meta");
    const path = fragment.querySelector(".library-path");
    const openChatButton = fragment.querySelector(".open-chat-button");
    const reindexProfile = fragment.querySelector(".reindex-profile");
    const reindexButton = fragment.querySelector(".reindex-button");
    const deleteButton = fragment.querySelector(".delete-button");
    const downloadJson = fragment.querySelector(".download-json");
    const downloadSrt = fragment.querySelector(".download-srt");

    title.textContent = video.title;
    meta.textContent = `${video.video_id} · ${video.status} · ${formatDuration(video.duration_seconds)} · ${video.language || "язык не определён"}`;
    path.textContent = video.source_path;
    openChatButton.disabled = video.status !== "indexed";
    openChatButton.addEventListener("click", () => {
      state.selectedVideoId = video.video_id;
      localStorage.setItem("selectedVideoId", state.selectedVideoId);
      populateVideoSelect();
      renderChat();
      renderSummary();
      selectTab("chat");
      setStatus(`Открыто видео: ${video.title}`);
    });

    reindexButton.addEventListener("click", async () => {
      setBusy(reindexButton, true);
      showActivity(`Переиндексирую видео: ${video.title}...`);
      try {
        await apiJson(`/api/videos/${encodeURIComponent(video.video_id)}/reindex`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ profile: reindexProfile.value }),
        });
        setStatus(`Переиндексация завершена: ${video.title}`);
        await refreshVideos();
        showActivity(`Переиндексация завершена: ${video.title}`, false);
      } catch (error) {
        hideActivity();
        setStatus(error.message, true);
      } finally {
        setBusy(reindexButton, false);
      }
    });

    deleteButton.addEventListener("click", async () => {
      const confirmed = window.confirm(`Удалить видео "${video.title}"?`);
      if (!confirmed) {
        return;
      }
      setBusy(deleteButton, true);
      showActivity(`Удаляю видео: ${video.title}...`);
      try {
        await apiJson(`/api/videos/${encodeURIComponent(video.video_id)}`, { method: "DELETE" });
        delete state.chatByVideo[video.video_id];
        persistChat();
        if (state.selectedVideoId === video.video_id) {
          state.selectedVideoId = "";
        }
        setStatus(`Видео удалено: ${video.title}`);
        await refreshVideos();
        showActivity(`Видео удалено: ${video.title}`, false);
      } catch (error) {
        hideActivity();
        setStatus(error.message, true);
      } finally {
        setBusy(deleteButton, false);
      }
    });

    downloadJson.href = `/api/videos/${encodeURIComponent(video.video_id)}/transcript/json`;
    downloadJson.setAttribute("download", `${video.video_id}.json`);
    downloadSrt.href = `/api/videos/${encodeURIComponent(video.video_id)}/transcript/srt`;
    downloadSrt.setAttribute("download", `${video.video_id}.srt`);

    card.dataset.videoId = video.video_id;
    elements.libraryList.appendChild(fragment);
  });
}

async function jumpToCitation(citation) {
  const video = getSelectedVideo();
  if (!video) {
    return;
  }
  try {
    const data = await apiJson(`/api/videos/${encodeURIComponent(video.video_id)}/jump/${encodeURIComponent(citation)}`);
    elements.videoPlayer.currentTime = Number(data.seconds || 0);
    elements.videoPlayer.play().catch(() => {});
  } catch (error) {
    setStatus(error.message, true);
  }
}

function buildHistoryPairs(chatItems) {
  const history = [];
  let pendingQuestion = null;
  chatItems.forEach((item) => {
    if (item.role === "user") {
      pendingQuestion = item.content;
      return;
    }
    if (item.role === "assistant" && pendingQuestion) {
      history.push({ question: pendingQuestion, answer: item.content });
      pendingQuestion = null;
    }
  });
  return history;
}

async function handleJson(response) {
  const contentType = response.headers.get("content-type") || "";
  const data = contentType.includes("application/json") ? await response.json() : null;
  if (!response.ok) {
    throw new Error(data?.detail || `HTTP ${response.status}`);
  }
  return data;
}

async function apiJson(url, options = {}) {
  const response = await fetch(url, options);
  return handleJson(response);
}

async function refreshVideos(options = {}) {
  const { silent = false } = options;
  if (state.refreshInFlight) {
    return;
  }
  state.refreshInFlight = true;
  if (!silent) {
    showActivity("Обновляю библиотеку и список видео...");
  }
  try {
    const videos = await apiJson("/api/videos");
    state.videos = videos;
    updateStats();
    populateVideoSelect();
    renderLibrary();
    renderChat();
    await renderSummary();
  } finally {
    state.refreshInFlight = false;
    if (!silent) {
      hideActivity();
    }
  }
}

function startAutoRefresh() {
  if (state.autoRefreshTimerId) {
    window.clearInterval(state.autoRefreshTimerId);
  }
  state.autoRefreshTimerId = window.setInterval(() => {
    if (document.hidden) {
      return;
    }
    refreshVideos({ silent: true }).catch((error) => {
      setStatus(error.message, true);
    });
  }, 10000);
}

async function syncOnVisibilityChange() {
  if (document.hidden) {
    return;
  }
  try {
    await refreshVideos({ silent: true });
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function submitQuestion(event) {
  event.preventDefault();
  const video = getSelectedVideo();
  const question = elements.questionInput.value.trim();
  if (!video) {
    setStatus("Сначала выберите проиндексированное видео.", true);
    return;
  }
  if (!question) {
    setStatus("Введите вопрос.", true);
    return;
  }

  const chatItems = getChatItems(video.video_id);
  chatItems.push({ role: "user", content: question });
  persistChat();
  renderChat();
  elements.questionInput.value = "";
  setBusy(elements.chatForm.querySelector("button[type='submit']"), true);
  setInlineActivity(elements.chatActivity, "Генерирую ответ по выбранному видео...");
  showActivity("Генерируется ответ в чате...");
  setStatus("Ищу ответ по видео...");

  try {
    const payload = {
      video_id: video.video_id,
      question,
      history: buildHistoryPairs(chatItems.slice(0, -1)),
    };
    const answer = await apiJson("/api/chat/answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    chatItems.push({ role: "assistant", content: answer.answer_text, citations: answer.citations || [] });
    persistChat();
    setInlineActivity(elements.chatActivity);
    renderChat();
    showActivity("Ответ в чате готов.", false);
    setStatus("Ответ готов.");
  } catch (error) {
    chatItems.pop();
    persistChat();
    setInlineActivity(elements.chatActivity);
    renderChat();
    hideActivity();
    setStatus(error.message, true);
  } finally {
    setInlineActivity(elements.chatActivity);
    setBusy(elements.chatForm.querySelector("button[type='submit']"), false);
  }
}

async function submitUpload(event) {
  event.preventDefault();
  const file = elements.uploadFile.files?.[0];
  if (!file) {
    setStatus("Выберите файл для загрузки.", true);
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  formData.append("profile", elements.uploadProfile.value);
  formData.append("language", elements.uploadLanguage.value.trim());

  const submitButton = elements.uploadForm.querySelector("button[type='submit']");
  setBusy(submitButton, true);
  showActivity("Видео загружается и индексируется. Это может занять время...");
  setStatus("Сохраняю файл и запускаю индексацию...");
  try {
    const video = await apiJson("/api/videos/upload", {
      method: "POST",
      body: formData,
    });
    state.selectedVideoId = video.video_id;
    localStorage.setItem("selectedVideoId", state.selectedVideoId);
    elements.uploadForm.reset();
    setStatus(`Индексация завершена: ${video.title}`);
    await refreshVideos();
    showActivity(`Индексация завершена: ${video.title}`, false);
    selectTab("chat");
  } catch (error) {
    hideActivity();
    setStatus(error.message, true);
  } finally {
    setBusy(submitButton, false);
  }
}

async function generateSummary() {
  const video = getSelectedVideo();
  if (!video) {
    setStatus("Сначала выберите видео.", true);
    return;
  }
  setBusy(elements.generateSummary, true);
  setInlineActivity(elements.summaryActivity, "Генерирую конспект по всему видео...");
  showActivity("Генерируется конспект...");
  setStatus("Генерирую конспект...");
  try {
    const data = await apiJson(`/api/videos/${encodeURIComponent(video.video_id)}/summary`, {
      method: "POST",
    });
    elements.summaryContent.className = "summary-content";
    renderMarkdown(elements.summaryContent, data.summary || "Конспект пуст.");
    setInlineActivity(elements.summaryActivity);
    showActivity("Конспект готов.", false);
    setStatus("Конспект готов.");
  } catch (error) {
    setInlineActivity(elements.summaryActivity);
    hideActivity();
    setStatus(error.message, true);
  } finally {
    setInlineActivity(elements.summaryActivity);
    setBusy(elements.generateSummary, false);
  }
}

function clearChat() {
  const video = getSelectedVideo();
  if (!video) {
    return;
  }
  state.chatByVideo[video.video_id] = [];
  persistChat();
  renderChat();
  setStatus("История очищена.");
}

function attachEvents() {
  elements.tabs.forEach((button) => {
    button.addEventListener("click", () => selectTab(button.dataset.tab));
  });
  elements.videoSelect.addEventListener("change", async (event) => {
    state.selectedVideoId = event.target.value;
    localStorage.setItem("selectedVideoId", state.selectedVideoId);
    renderChat();
    await renderSummary();
  });
  document.addEventListener("visibilitychange", syncOnVisibilityChange);
  window.addEventListener("focus", syncOnVisibilityChange);
  elements.chatForm.addEventListener("submit", submitQuestion);
  elements.uploadForm.addEventListener("submit", submitUpload);
  elements.generateSummary.addEventListener("click", generateSummary);
  elements.clearChat.addEventListener("click", clearChat);
}

async function init() {
  attachEvents();
  startAutoRefresh();
  selectTab(state.activeTab);
  setStatus("Загружаю библиотеку...");
  try {
    await refreshVideos();
    setStatus("Готово.");
  } catch (error) {
    setStatus(error.message, true);
  }
}

init();
