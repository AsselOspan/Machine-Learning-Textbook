(() => {
  "use strict";
  const themeButton = document.querySelector("[data-theme-toggle]");
  const menuButton = document.querySelector("[data-menu-toggle]");
  const sidebar = document.querySelector(".sidebar");
  const status = document.getElementById("action-status");
  const announce = message => { if (status) status.textContent = message; };

  function setTheme(theme) {
    document.documentElement.dataset.theme = theme;
    if (themeButton) {
      themeButton.textContent = theme === "dark" ? "Жарық тақырып" : "Қараңғы тақырып";
      themeButton.setAttribute("aria-pressed", String(theme === "dark"));
    }
    try { localStorage.setItem("ml-kk-theme", theme); } catch (_) {}
  }
  let savedTheme;
  try { savedTheme = localStorage.getItem("ml-kk-theme"); } catch (_) {}
  setTheme(savedTheme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
  themeButton?.addEventListener("click", () => {
    setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
  });

  function closeMenu() {
    sidebar?.classList.remove("open");
    menuButton?.setAttribute("aria-expanded", "false");
  }
  menuButton?.addEventListener("click", () => {
    const open = sidebar.classList.toggle("open");
    menuButton.setAttribute("aria-expanded", String(open));
  });
  document.addEventListener("keydown", event => {
    if (event.key === "Escape" && sidebar?.classList.contains("open")) {
      closeMenu();
      menuButton?.focus();
    }
  });
  sidebar?.addEventListener("click", event => {
    if (event.target.closest("a") && matchMedia("(max-width:760px)").matches) closeMenu();
  });

  const prefix = document.body.dataset.root || "";
  const search = document.getElementById("search-input");
  const results = document.getElementById("search-results");
  const searchStatus = document.getElementById("search-status");
  const normalize = value => value.normalize("NFKC").toLocaleLowerCase("kk-KZ").trim();
  search?.addEventListener("input", () => {
    results.replaceChildren();
    const query = normalize(search.value);
    if (query.length < 2) {
      searchStatus.textContent = "Іздеу үшін кемінде екі таңба енгізіңіз.";
      return;
    }
    const matches = (window.ML_SEARCH || []).filter(item => normalize(item.text).includes(query));
    searchStatus.textContent = matches.length ? "Табылған бөлімдер — " + matches.length + "." : "Сәйкес бөлім табылмады.";
    matches.slice(0, 20).forEach(item => {
      const row = document.createElement("li");
      const link = document.createElement("a");
      link.href = prefix + item.url;
      link.textContent = item.title;
      const snippet = document.createElement("p");
      const position = normalize(item.text).indexOf(query);
      const start = Math.max(0, position - 65);
      snippet.textContent = item.text.slice(start, start + 220);
      row.append(link, snippet);
      results.append(row);
    });
  });
  document.querySelector("[data-open-search]")?.addEventListener("click", () => {
    search?.scrollIntoView({ block: "center", behavior: "smooth" });
    search?.focus({ preventScroll: true });
  });

  document.querySelectorAll("pre").forEach(pre => {
    const container = document.createElement("div");
    container.className = "code-container";
    pre.before(container);
    container.append(pre);
    const button = document.createElement("button");
    button.className = "copy-code";
    button.type = "button";
    button.textContent = "Кодты көшіру";
    button.addEventListener("click", async () => {
      const text = pre.textContent;
      let copied = false;
      try {
        await navigator.clipboard.writeText(text);
        copied = true;
      } catch (_) {
        const temporary = document.createElement("textarea");
        temporary.value = text;
        temporary.style.position = "fixed";
        temporary.style.left = "-9999px";
        document.body.append(temporary);
        temporary.select();
        try { copied = document.execCommand("copy"); } catch (_) {}
        temporary.remove();
        button.focus();
      }
      button.textContent = copied ? "Көшірілді" : "Мәтінді таңдап көшіріңіз";
      announce(copied ? "Код алмасу буферіне көшірілді." : "Браузер көшіруді шектеді.");
    });
    container.append(button);
  });

  document.querySelectorAll("[data-linear-demo]").forEach(demo => {
    const slider = demo.querySelector("input");
    const line = demo.querySelector("[data-prediction-line]");
    const residuals = demo.querySelector("[data-residuals]");
    const output = demo.querySelector("output");
    function update() {
      const weight = Number(slider.value);
      const x = [0, 1, 2];
      const y = [1, 3, 5];
      const prediction = x.map(value => 1 + weight * value);
      const mse = y.reduce((sum, value, index) => sum + (value - prediction[index]) ** 2, 0) / y.length;
      const py = value => 185 - value * 25;
      line.setAttribute("y1", py(prediction[0]));
      line.setAttribute("y2", py(prediction[2]));
      residuals.replaceChildren();
      x.forEach((value, index) => {
        const segment = document.createElementNS("http://www.w3.org/2000/svg", "line");
        segment.setAttribute("x1", 50 + value * 145);
        segment.setAttribute("x2", 50 + value * 145);
        segment.setAttribute("y1", py(y[index]));
        segment.setAttribute("y2", py(prediction[index]));
        segment.setAttribute("stroke", "#c17a32");
        segment.setAttribute("stroke-dasharray", "4 3");
        residuals.append(segment);
      });
      output.textContent = "w = " + weight.toFixed(2) + ", MSE = " + mse.toFixed(4);
      demo.dataset.mse = String(mse);
    }
    slider.addEventListener("input", update);
    update();
  });
})();
