() => {
  // Assistente Refauna — comportamento da interface. Sem dependências externas; tudo por delegação de eventos.
  const THEME_KEY = "refauna-theme";
  const SIDEBAR_KEY = "refauna-sidebar";
  const store = {
    get(k) { try { return window.localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { window.localStorage.setItem(k, v); } catch (e) { /* modo privado: segue sem salvar */ } },
  };
  const body = document.body;
  const mq = window.matchMedia("(prefers-color-scheme: dark)");

  // ---- tema: começa no do sistema; a escolha manual fica salva no navegador
  const wantDark = () => (store.get(THEME_KEY) || (mq.matches ? "dark" : "light")) === "dark";
  const applyTheme = () => {
    const dark = wantDark();
    if (body.classList.contains("dark") !== dark) body.classList.toggle("dark", dark);
    document.documentElement.style.colorScheme = dark ? "dark" : "light";
  };
  applyTheme();
  // O Gradio também mexe na classe "dark" (tema do sistema); reaplica a escolha do usuário.
  new MutationObserver(applyTheme).observe(body, { attributes: true, attributeFilter: ["class"] });
  mq.addEventListener("change", applyTheme);
  const toggleTheme = () => { store.set(THEME_KEY, wantDark() ? "light" : "dark"); applyTheme(); };

  // ---- barra lateral recolhível (estado salvo; em tela estreita começa fechada)
  const setSidebar = (collapsed, save) => {
    body.classList.toggle("rf-collapsed", collapsed);
    if (save) store.set(SIDEBAR_KEY, collapsed ? "1" : "0");
  };
  const savedSb = store.get(SIDEBAR_KEY);
  setSidebar(savedSb === "1" || (savedSb === null && window.innerWidth < 900), false);

  // ---- visões: Chat · Comparar · Resultados (sem ida ao servidor)
  window.rfView = (v) => {
    body.dataset.rfView = v;
    document.querySelectorAll("[data-rf-nav]").forEach((b) => {
      const on = b.dataset.rfNav === v;
      b.classList.toggle("active", on);
      b.setAttribute("aria-selected", String(on));
    });
    if (window.innerWidth < 760) setSidebar(true, false);
  };
  window.rfView("chat");

  // ---- estado "conversa vazia" (centraliza o campo como na tela inicial)
  // + pergunta nova: rola até o fim mesmo se o usuário tinha subido para ler as fontes
  let userRows = 0;
  const syncEmpty = () => {
    const bot = document.getElementById("rf-chatbot");
    const empty = !bot || !bot.querySelector(".message-row, .message");
    body.classList.toggle("rf-empty", empty);
    const n = bot ? bot.querySelectorAll(".user-row").length : 0;
    if (n > userRows) {
      const wrap = bot.querySelector(".bubble-wrap");
      if (wrap) requestAnimationFrame(() => { wrap.scrollTop = wrap.scrollHeight; });
    }
    userRows = n;
    // título da conversa = 1ª pergunta visível (redundante com o servidor, mas à prova de corrida)
    const first = bot && bot.querySelector(".user-row");
    const titleEl = document.querySelector("#rf-title .rf-title-text");
    if (first && titleEl) {
      const txt = first.innerText.trim();
      const short = txt.length > 60 ? txt.slice(0, 60).trimEnd() + "…" : txt;
      if (titleEl.textContent !== short) titleEl.textContent = short;
    }
    // gerando = botão Parar visível: bloqueia novas perguntas até terminar (evita respostas órfãs)
    const stop = document.getElementById("rf-stop");
    body.classList.toggle("rf-busy", !!stop && stop.offsetParent !== null);
  };
  document.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && body.classList.contains("rf-busy") && e.target.closest("#rf-input")) {
      e.preventDefault();
      e.stopPropagation();
    }
  }, true);
  syncEmpty();
  new MutationObserver(syncEmpty).observe(document.getElementById("rf-main") || body, { childList: true, subtree: true });
  const dock = document.getElementById("rf-dock");
  if (dock) new MutationObserver(syncEmpty).observe(dock, { attributes: true, subtree: true, attributeFilter: ["class", "style", "hidden"] });

  // ---- seletor de modelo
  const picker = () => document.getElementById("rf-picker");
  const closeMenu = () => {
    const p = picker();
    if (!p) return;
    p.classList.remove("open");
    p.querySelector(".rf-picker-btn")?.setAttribute("aria-expanded", "false");
  };
  const selectModel = (item) => {
    const key = item.dataset.rfModel;
    const radio = Array.from(document.querySelectorAll("#rf-model input[type=radio]"))
      .find((r) => r.value === key || r.closest("label")?.textContent.trim() === key);
    if (!radio) return;
    radio.click();
    const p = picker();
    p.querySelector(".rf-picker-label").textContent = item.dataset.label;
    p.querySelectorAll(".rf-model-item").forEach((b) => {
      const on = b === item;
      b.classList.toggle("is-current", on);
      b.setAttribute("aria-checked", String(on));
    });
    closeMenu();
  };

  // ---- citações [n]: abre "Fontes" da mesma resposta e destaca o trecho n
  const openSource = (cite) => {
    const n = cite.textContent.trim();
    let box = cite.parentElement;
    while (box && box !== document.body && !box.querySelector("details.rf-sources")) box = box.parentElement;
    const det = box && box.querySelector("details.rf-sources");
    if (!det) return;
    det.open = true;
    const li = det.querySelector(`li.rf-s${n}`);
    if (!li) return;
    requestAnimationFrame(() => li.scrollIntoView({ behavior: "smooth", block: "center" }));
    li.classList.remove("rf-flash");
    void li.offsetWidth;
    li.classList.add("rf-flash");
  };

  document.addEventListener("click", (e) => {
    const t = e.target.closest("[data-rf-act], [data-rf-nav], [data-rf-model], .rf-cite");
    if (!t) {
      if (!e.target.closest("#rf-picker")) closeMenu();
      return;
    }
    if (t.classList.contains("rf-cite")) { e.preventDefault(); e.stopPropagation(); openSource(t); return; }
    if (t.dataset.rfModel) { if (t.getAttribute("aria-disabled") !== "true") selectModel(t); return; }
    if (t.dataset.rfNav) { window.rfView(t.dataset.rfNav); closeMenu(); return; }
    const act = t.dataset.rfAct;
    if (act === "theme") toggleTheme();
    else if (act === "sidebar") setSidebar(!body.classList.contains("rf-collapsed"), true);
    else if (act === "examples") {
      const ex = document.getElementById("rf-examples");
      const closed = ex.classList.toggle("rf-closed");
      t.setAttribute("aria-expanded", String(!closed));
    } else if (act === "model-menu") {
      const p = picker();
      const open = p.classList.toggle("open");
      t.setAttribute("aria-expanded", String(open));
    }
  }, true);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeMenu(); });
}
