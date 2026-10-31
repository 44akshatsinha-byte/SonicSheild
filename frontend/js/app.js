(() => {
  const views = ["home", "analyze", "insights", "evaluation", "history"];
  const crumb = document.getElementById("crumbSection");
  const nav = document.getElementById("nav");
  const clockEl = document.getElementById("clock");

  const splitHeadings = () => {
    document.querySelectorAll(".split").forEach((el) => {
      if (el.dataset.split) return;
      const text = el.textContent.trim();
      el.textContent = "";
      text.split(" ").forEach((word) => {
        const w = document.createElement("span");
        w.style.display = "inline-block";
        w.style.overflow = "hidden";
        w.style.verticalAlign = "bottom";
        w.style.marginRight = "0.28em";
        const inner = document.createElement("span");
        inner.style.display = "inline-block";
        inner.textContent = word;
        w.appendChild(inner);
        el.appendChild(w);
        if (window.gsap) {
          gsap.from(inner, {
            yPercent: 110,
            duration: 1.1,
            ease: "power3.out",
            delay: 0.15,
          });
        }
      });
      el.dataset.split = "1";
    });
  };

  const setView = (name, scrollId) => {
    views.forEach((id) => {
      const node = document.getElementById(`view-${id}`);
      if (node) node.classList.toggle("active", id === name);
    });
    crumb.textContent = name === "home" ? "Welcome" : name.replace("-", " ");
    nav.classList.remove("open");
    document.body.style.overflow = "";
    window.location.hash = name === "home" ? "home" : name;
    if (name === "home" && window.ScrollTrigger) {
      requestAnimationFrame(() => ScrollTrigger.refresh());
    }
    if (scrollId) {
      const target = document.getElementById(scrollId);
      if (target) target.scrollIntoView({ behavior: "smooth" });
    } else {
      window.scrollTo({ top: 0, behavior: "instant" });
    }
    if (name === "history") renderHistory();
  };

  document.querySelectorAll("[data-view]").forEach((el) => {
    el.addEventListener("click", (e) => {
      e.preventDefault();
      setView(el.dataset.view, el.dataset.scroll);
    });
  });

  document.getElementById("menuOpen").addEventListener("click", () => {
    nav.classList.add("open");
    document.body.style.overflow = "hidden";
  });
  document.getElementById("menuClose").addEventListener("click", () => {
    nav.classList.remove("open");
    document.body.style.overflow = "";
  });

  const tickClock = () => {
    const now = new Date();
    const fmt = new Intl.DateTimeFormat("en-GB", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: true,
    }).format(now);
    clockEl.textContent = `Local ${fmt}`;
  };
  tickClock();
  setInterval(tickClock, 30000);

  document.querySelectorAll(".acc-item button").forEach((btn) => {
    btn.addEventListener("click", () => {
      const item = btn.parentElement;
      const group = item.parentElement;
      group.querySelectorAll(".acc-item").forEach((n) => {
        if (n !== item) n.classList.remove("open");
      });
      item.classList.toggle("open");
    });
  });

  if (window.Lenis && window.gsap && window.ScrollTrigger) {
    gsap.registerPlugin(ScrollTrigger);
    const lenis = new Lenis({ lerp: 0.08, smoothWheel: true });
    lenis.on("scroll", ScrollTrigger.update);
    gsap.ticker.add((time) => lenis.raf(time * 1000));
    gsap.ticker.lagSmoothing(0);

    const track = document.getElementById("highlightsTrack");
    if (track) {
      ScrollTrigger.create({
        trigger: "#highlights",
        start: "top top",
        end: () => `+=${track.scrollWidth}`,
        pin: true,
        scrub: 1,
        anticipatePin: 1,
        onUpdate: (self) => {
          const x = -self.progress * (track.scrollWidth - window.innerWidth);
          track.style.transform = `translate3d(${x}px,0,0)`;
        },
      });
    }

    const steps = document.querySelectorAll(".method-step");
    steps.forEach((step, i) => {
      ScrollTrigger.create({
        trigger: "#methodology",
        start: () => `top+=${i * 80} center`,
        onEnter: () => {
          steps.forEach((s) => s.classList.remove("active"));
          step.classList.add("active");
        },
        onEnterBack: () => {
          steps.forEach((s) => s.classList.remove("active"));
          step.classList.add("active");
        },
      });
    });

    ScrollTrigger.create({
      trigger: "#welcome",
      start: "top top",
      end: "bottom top",
      onEnter: () => (crumb.textContent = "Welcome"),
      onEnterBack: () => (crumb.textContent = "Welcome"),
    });
    ScrollTrigger.create({
      trigger: "#highlights",
      start: "top center",
      onEnter: () => (crumb.textContent = "Highlights"),
      onEnterBack: () => (crumb.textContent = "Highlights"),
    });
    ScrollTrigger.create({
      trigger: "#expertise",
      start: "top center",
      onEnter: () => (crumb.textContent = "Expertise"),
      onEnterBack: () => (crumb.textContent = "Expertise"),
    });
    ScrollTrigger.create({
      trigger: "#methodology",
      start: "top center",
      onEnter: () => (crumb.textContent = "Methodology"),
      onEnterBack: () => (crumb.textContent = "Methodology"),
    });
  }

  splitHeadings();

  const hash = (window.location.hash || "#home").replace("#", "");
  if (views.includes(hash)) setView(hash);

  fetch("/health")
    .then((r) => r.json())
    .then((data) => {
      const aasist = data.aasist_model?.loaded ? "AASIST live" : "AASIST unloaded";
      document.getElementById("navHealth").textContent = `${data.status} · ${aasist}`;
    })
    .catch(() => {
      document.getElementById("navHealth").textContent = "API offline — start uvicorn";
    });

  const fileInput = document.getElementById("fileInput");
  const dropzone = document.getElementById("dropzone");
  const fileLabel = document.getElementById("fileLabel");
  const sampleSelect = document.getElementById("sampleSelect");
  let chosenFile = null;

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("drag");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag");
    if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) setFile(fileInput.files[0]);
  });
  const setFile = (file) => {
    chosenFile = file;
    fileLabel.textContent = file.name;
    sampleSelect.value = "";
  };

  fetch("/samples")
    .then((r) => r.json())
    .then((data) => {
      (data.samples || []).forEach((s) => {
        const opt = document.createElement("option");
        opt.value = s.filename;
        opt.textContent = s.name.replace(/_/g, " ");
        sampleSelect.appendChild(opt);
      });
    })
    .catch(() => {});

  sampleSelect.addEventListener("change", async () => {
    if (!sampleSelect.value) return;
    const res = await fetch(`/samples/${encodeURIComponent(sampleSelect.value)}`);
    const blob = await res.blob();
    chosenFile = new File([blob], sampleSelect.value, { type: "audio/wav" });
    fileLabel.textContent = sampleSelect.value;
  });

  const verdictCopy = (data) => {
    const mode = data.detection_mode;
    const cls = data.overall_aasist_class;
    const score = data.overall_liveness ?? 0;
    if (mode === "pretrained" && cls === "bona_fide") {
      return ["Model leans bona-fide", "Spectro-temporal features align more closely with live speech."];
    }
    if (mode === "pretrained" && cls === "spoof") {
      return ["Model leans synthetic", "Anti-spoofing patterns associated with cloned or generated audio."];
    }
    if (score >= 70) return ["Model leans bona-fide", "Room and vocal indicators sit in expected live ranges."];
    if (score <= 40) return ["Model leans synthetic", "Room or vocal dynamics look unnatural under this pass."];
    return ["Inconclusive", "Metrics sit in an intermediate zone. Human review recommended."];
  };

  document.getElementById("runBtn").addEventListener("click", async () => {
    const status = document.getElementById("status");
    if (!chosenFile) {
      status.textContent = "Choose a WAV file first.";
      return;
    }
    status.textContent = "Running pipeline…";
    const params = new URLSearchParams({
      codec: document.getElementById("codec").value,
      mode: document.getElementById("mode").value,
      detection_mode: document.getElementById("detection").value,
    });
    const body = new FormData();
    body.append("file", chosenFile, chosenFile.name);
    try {
      const res = await fetch(`/analyze?${params.toString()}`, { method: "POST", body });
      const data = await res.json();
      if (!res.ok) {
        status.textContent = data.detail || "Analysis failed.";
        return;
      }
      status.textContent = `Done in ${data.processing_time_ms} ms · ${data.num_windows} windows`;
      const [title, desc] = verdictCopy(data);
      document.getElementById("verdict").innerHTML = `<h3 class="display">${title}</h3><p>${desc}</p>`;
      document.getElementById("metrics").innerHTML = [
        ["Liveness", data.overall_liveness],
        ["Baseline", data.overall_baseline_liveness ?? "—"],
        ["AASIST", data.overall_aasist_liveness ?? "—"],
        ["ML", data.overall_ml_liveness ?? "—"],
        ["Physics", data.physics_mean],
        ["Biology", data.biology_mean],
        ["Band", data.overall_band_label || data.overall_band],
        ["Codec", data.codec],
      ]
        .map(([k, v]) => `<div class="metric"><span>${k}</span><b>${v ?? "—"}</b></div>`)
        .join("");
      const timeline = document.getElementById("timeline");
      timeline.innerHTML = "";
      (data.windows || []).forEach((w) => {
        const bar = document.createElement("i");
        const val = w.liveness ?? w.baseline_score ?? 0;
        bar.style.height = `${Math.max(8, val)}%`;
        timeline.appendChild(bar);
      });
      document.getElementById("results").classList.remove("hidden");
      const log = JSON.parse(localStorage.getItem("ss-history") || "[]");
      log.unshift({
        t: new Date().toISOString(),
        file: chosenFile.name,
        mode: data.detection_mode,
        score: data.overall_liveness,
        band: data.overall_band,
      });
      localStorage.setItem("ss-history", JSON.stringify(log.slice(0, 40)));
    } catch (err) {
      status.textContent = `Request failed: ${err.message}`;
    }
  });

  const renderHistory = () => {
    const log = JSON.parse(localStorage.getItem("ss-history") || "[]");
    const body = document.getElementById("historyBody");
    if (!log.length) {
      body.innerHTML = `<tr><td colspan="5">No inspections in this browser session.</td></tr>`;
      return;
    }
    body.innerHTML = log
      .map(
        (row) =>
          `<tr><td>${row.t.replace("T", " ").slice(0, 19)}</td><td>${row.file}</td><td>${row.mode}</td><td>${row.score}</td><td>${row.band}</td></tr>`
      )
      .join("");
  };
})();
