(() => {
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
  const pauseFor = 4800;

  for (const lesson of document.querySelectorAll("[data-visual-lesson]")) {
    const image = lesson.querySelector(".visual-frame");
    const frames = JSON.parse(lesson.querySelector("[data-visual-frames]").textContent);
    const stage = lesson.querySelector(".visual-stage");
    const title = stage.querySelector("strong");
    const detail = stage.querySelector("p");
    const controls = lesson.querySelector(".visual-controls");
    const previous = controls.querySelector("[data-visual-prev]");
    const next = controls.querySelector("[data-visual-next]");
    const play = controls.querySelector("[data-visual-play]");
    const count = controls.querySelector(".visual-count");
    let position = 0;
    let timer = null;
    let imagesReady = false;

    function preloadFrames() {
      if (imagesReady) return;
      imagesReady = true;
      for (const frame of frames.slice(1)) {
        const preview = new Image();
        preview.src = frame.src;
      }
    }

    function stop() {
      clearTimeout(timer);
      timer = null;
      play.textContent = "Ойнату";
      stage.setAttribute("aria-live", "polite");
    }

    function show(index, animate) {
      position = index;
      const frame = frames[index];
      image.classList.remove("is-changing");
      image.src = frame.src;
      image.alt = frame.label;
      if (animate && !reducedMotion.matches) {
        void image.offsetWidth;
        image.classList.add("is-changing");
      }
      title.textContent = frame.label;
      detail.textContent = frame.detail;
      count.textContent = `${index + 1} / ${frames.length}`;
      previous.disabled = index === 0;
      next.disabled = index === frames.length - 1;
    }

    function advance() {
      timer = setTimeout(() => {
        if (position === frames.length - 1 || document.hidden) {
          stop();
          return;
        }
        show(position + 1, true);
        if (position === frames.length - 1) stop();
        else advance();
      }, pauseFor);
    }

    function moveBy(amount) {
      stop();
      preloadFrames();
      show(Math.max(0, Math.min(frames.length - 1, position + amount)), true);
    }

    previous.addEventListener("click", () => moveBy(-1));
    next.addEventListener("click", () => moveBy(1));
    play.addEventListener("click", () => {
      if (timer !== null) {
        stop();
        return;
      }
      preloadFrames();
      if (position === frames.length - 1) show(0, true);
      stage.setAttribute("aria-live", "off");
      play.textContent = "Кідірту";
      advance();
    });
    controls.addEventListener("keydown", event => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      moveBy(event.key === "ArrowLeft" ? -1 : 1);
    });
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) stop();
    });
    controls.hidden = false;
    show(0, false);
  }
})();
