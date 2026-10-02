const thread = document.getElementById("thread");
const form = document.getElementById("form");
const input = document.getElementById("input");
const send = document.getElementById("send");
const health = document.getElementById("health");
const history = [];

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderEmpty() {
  thread.innerHTML =
    '<p class="empty">Aucune question pour l’instant. Choisissez une suggestion ou écrivez dans le champ ci-dessous.</p>';
}

function appendBubble(role, text, tools = []) {
  const empty = thread.querySelector(".empty");
  if (empty) empty.remove();
  const wrap = document.createElement("div");
  wrap.className = `bubble ${role}`;
  wrap.innerHTML = escapeHtml(text);
  if (tools.length) {
    const bar = document.createElement("div");
    bar.className = "tools";
    bar.innerHTML = tools
      .map((tool) => `<span class="tool">${escapeHtml(tool.name)}</span>`)
      .join("");
    wrap.appendChild(bar);
  }
  thread.appendChild(wrap);
  thread.scrollTop = thread.scrollHeight;
  return wrap;
}

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.ok) {
      health.textContent = "Agent prêt";
      health.className = "status ok";
    } else {
      health.textContent = data.error || "Agent indisponible (clé OpenAI ?)";
      health.className = "status bad";
    }
  } catch {
    health.textContent = "Serveur injoignable";
    health.className = "status bad";
  }
}

async function ask(question) {
  history.push({ role: "user", content: question });
  appendBubble("user", question);
  const pending = appendBubble("assistant", "Recherche en cours…");
  send.disabled = true;
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Erreur serveur");
    }
    pending.textContent = data.answer;
    if (data.tools?.length) {
      const bar = document.createElement("div");
      bar.className = "tools";
      bar.innerHTML = data.tools
        .map((tool) => `<span class="tool">${escapeHtml(tool.name)}</span>`)
        .join("");
      pending.appendChild(bar);
    }
    history.push({ role: "assistant", content: data.answer });
  } catch (err) {
    pending.classList.add("error");
    pending.textContent = err.message;
    history.pop();
  } finally {
    send.disabled = false;
    thread.scrollTop = thread.scrollHeight;
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  input.value = "";
  ask(question);
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.getElementById("suggestions").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-q]");
  if (button) ask(button.dataset.q);
});

document.getElementById("clear").addEventListener("click", () => {
  history.length = 0;
  renderEmpty();
});

renderEmpty();
checkHealth();
